from .vcc import VCCBase
from semilearn.algorithms.simmatch import SimMatch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import torch
from semilearn.algorithms.utils import ce_loss, consistency_loss
from ..simmatch.simmatch import SimMatch_Net


class VCCSimMatch(VCCBase, SimMatch):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        SimMatch.__init__(self, args, net_builder, tb_log, logger)
        VCCBase.__init__(self, args, net_builder, tb_log, logger)

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes, args=self.args)
        ema_model = SimMatch_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = len(x_ulb_w['input_ids']) if isinstance(x_ulb_w, dict) else x_ulb_w.shape[0]
        idx_lb = idx_lb.cuda(self.gpu)

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            bank = self.mem_bank.clone().detach()
            assert self.use_cat
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            outputs = self.model(inputs)
            logits, feats = outputs['logits'], outputs['feat']
            logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
            ema_logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            ema_feats_x_ulb_w, feats_x_ulb_s = feats[num_lb:].chunk(2)
            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            self.ema.apply_shadow()
            with torch.no_grad():
                # ema teacher model
                if self.use_ema_teacher:
                    ema_feats_x_lb = self.model(x_lb)['feat']
                ema_probs_x_ulb_w = F.softmax(ema_logits_x_ulb_w, dim=-1)
                ema_probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook",
                                                   probs_x_ulb=ema_probs_x_ulb_w.detach())
            self.ema.restore()

            with torch.no_grad():
                teacher_logits = ema_feats_x_ulb_w @ bank
                teacher_prob_orig = F.softmax(teacher_logits / self.T, dim=1)
                factor = ema_probs_x_ulb_w.gather(1, self.labels_bank.expand([num_ulb, -1]))
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if self.smoothing_alpha < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, self.labels_bank.expand([bs, -1]),
                                                                  teacher_prob_orig)
                    probs_x_ulb_w = ema_probs_x_ulb_w * self.smoothing_alpha + aggregated_prob * (
                                1 - self.smoothing_alpha)
                else:
                    probs_x_ulb_w = ema_probs_x_ulb_w

            student_logits = feats_x_ulb_s @ bank
            student_prob = F.softmax(student_logits / self.T, dim=1)
            in_loss = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1).mean()
            if self.epoch == 0:
                in_loss *= 0.0
                probs_x_ulb_w = ema_probs_x_ulb_w

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          probs_x_ulb_w,
                                          'ce',
                                          mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_in * in_loss

            self.update_bank(ema_feats_x_lb, y_lb, idx_lb)

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict = self.update_save_dict(save_dict)
        return save_dict

    def load_model(self, load_path):
        checkpoint = super(VCC, self).load_model(load_path)
        checkpoint = self.load_vcc_model(checkpoint)
        return checkpoint