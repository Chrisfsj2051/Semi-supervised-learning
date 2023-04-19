from itertools import chain

from .vcc import VCCBase
from semilearn.algorithms.simmatch import SimMatch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
from semilearn.algorithms.utils import ce_loss, consistency_loss
from ..simmatch.simmatch import SimMatch_Net
from ...core import AlgorithmBase


class VCC_SimMatch_Net(SimMatch_Net):

    def forward(self, algorithm, x, idx_ulb=None, **kwargs):
        outs = self.backbone(algorithm, x, ulb_x_idx=idx_ulb)
        feat, logits = outs['feats'], outs['logits']
        feat_proj = self.l2norm(self.mlp_proj(feat))
        outs['feat'] = feat_proj
        return outs


class VCCSimMatch(VCCBase, SimMatch):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        SimMatch.__init__(self, args, net_builder, tb_log, logger)
        VCCBase.__init__(self, args, net_builder, tb_log, logger)

    def set_model(self):
        model = AlgorithmBase.set_model(self)
        model = VCC_SimMatch_Net(model, proj_size=self.args.proj_size)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes, args=self.args)
        ema_model = VCC_SimMatch_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def update_save_dict(self, save_dict):
        save_dict['uncertainty_selected'] = self.uncertainty_selected.cpu()
        save_dict = self.model.module.backbone.update_save_dict(save_dict)
        return save_dict

    def load_vcc_model(self, checkpoint):
        self.uncertainty_selected = checkpoint['uncertainty_selected'].cuda(self.gpu)
        self.model.module.backbone.load_model(checkpoint)
        self.print_fn("additional VCC parameter loaded")
        return checkpoint

    def set_vcc_requires_grad(self, requires_grad):
        params_list = chain(self.model.module.backbone.encoder.parameters(),
                            self.model.module.backbone.decoder.parameters())
        for param in params_list:
            param.requires_grad = requires_grad

    def train_step(self, idx_lb, idx_ulb, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = len(x_ulb_w['input_ids']) if isinstance(x_ulb_w, dict) else x_ulb_w.shape[0]
        idx_lb = idx_lb.cuda(self.gpu)

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            bank = self.mem_bank.clone().detach()
            assert self.use_cat
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            outputs = self.model(self, inputs, idx_ulb)
            logits, feats = outputs['logits'], outputs['feat']
            logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
            ema_logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            ema_feats_x_ulb_w, feats_x_ulb_s = feats[num_lb:].chunk(2)

            logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            recon_pred_ulb_w, recon_pred_ulb_s = outputs['recon_pred'][num_lb:].chunk(2)
            recon_gt_ulb_w, recon_gt_ulb_s = outputs['recon_gt'][num_lb:].chunk(2)
            mu_ulb_w, mu_ulb_s = outputs['mu'][num_lb:].chunk(2)
            logvar_ulb_w, logvar_ulb_s = outputs['logvar'][num_lb:].chunk(2)
            recon_pred_lb, recon_gt_lb = outputs['recon_pred'][:num_lb], outputs['recon_gt'][:num_lb]
            mu_lb, logvar_lb = outputs['mu'][:num_lb], outputs['logvar'][:num_lb]
            calibrated_logits_ulb_w, calibrated_logits_ulb_s = outputs['calibrated_logits'][num_lb:].chunk(2)

            self.ema.apply_shadow()
            with torch.no_grad():
                # ema teacher model
                if self.use_ema_teacher:
                    ema_feats_x_lb = self.model(x_lb)['feat']
                ema_probs_x_ulb_w = F.softmax(ema_logits_x_ulb_w, dim=-1)
            self.ema.restore()

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            vcc_probs = ema_probs_x_ulb_w
            vcc_unlab_recon_loss_weight, vcc_unlab_kl_loss_weight = 0.0, 0.0
            vcc_lab_loss_weight = 0.0

            if self.it < self.vcc_training_warmup:
                self.set_vcc_requires_grad(False)
            elif self.it == self.vcc_training_warmup:
                self.set_vcc_requires_grad(True)

            if self.it > self.vcc_training_warmup:
                loss_warmup_alpha = min((self.it - self.vcc_training_warmup) / 100, 1.0)
                vcc_unlab_recon_loss_weight = self.vcc_unlab_recon_loss_weight * loss_warmup_alpha
                vcc_unlab_kl_loss_weight = self.vcc_unlab_kl_loss_weight * loss_warmup_alpha
                vcc_lab_loss_weight = self.vcc_lab_loss_weight * loss_warmup_alpha
                self.p_cutoff = self.args.p_cutoff
            if self.it > self.vcc_selection_warmup:
                if self.args.vcc_disable_variance or self.it < self.args.vcc_variance_warmup:
                    vcc_probs = recon_gt_ulb_w
                else:
                    vcc_probs = F.softmax(calibrated_logits_ulb_w, dim=-1)
                self.p_cutoff = self.args.vcc_p_cutoff

            vcc_probs = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=vcc_probs.detach())
            ema_probs_x_ulb_w = vcc_probs

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

            vcc_loss_ulb_w = self.compute_vcc_loss(recon_pred_ulb_w, recon_gt_ulb_w, logvar_ulb_w, mu_ulb_w, mask)
            recon_loss_ulb_w = vcc_loss_ulb_w['recon_loss'] * vcc_unlab_recon_loss_weight
            kl_loss_ulb_w = vcc_loss_ulb_w['kl_loss'] * vcc_unlab_kl_loss_weight
            vcc_loss_lb = self.compute_vcc_loss(recon_pred_lb, recon_gt_lb, logvar_lb, mu_lb,
                                                mask.new_ones(recon_pred_lb.shape[0]))
            recon_loss_lb = vcc_loss_lb['recon_loss'] * vcc_lab_loss_weight
            kl_loss_lb = vcc_loss_lb['kl_loss'] * vcc_lab_loss_weight

            total_loss = (sup_loss + self.lambda_u * unsup_loss + self.lambda_in * in_loss +
                          recon_loss_ulb_w + kl_loss_ulb_w + kl_loss_lb + recon_loss_lb)

            self.update_bank(ema_feats_x_lb, y_lb, idx_lb)

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/kl_unsup_loss'] = kl_loss_ulb_w.item()
        tb_dict['train/recon_unsup_loss'] = recon_loss_ulb_w.item()
        tb_dict['train/kl_sup_loss'] = kl_loss_lb.item()
        tb_dict['train/recon_sup_loss'] = recon_loss_lb.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict = self.update_save_dict(save_dict)
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        checkpoint = self.load_vcc_model(checkpoint)
        return checkpoint