from semilearn.algorithms import FixMatch
import torch.nn as nn
import torch

from semilearn.algorithms.utils import consistency_loss, ce_loss


class MLPFixMatch(FixMatch):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.calc_weight_method = args.calc_weight_method
        self.l2_weight = args.l2_weight
        self.model.select_model = nn.Sequential(
            nn.Linear(self.model.channels, 1)
        )

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            outputs = self.model(inputs)
            logits_x_lb = outputs['logits'][:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            feat_x_ulb_w, feat_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            example_logits_ulb_w, example_logits_ulb_s = outputs['example_logits'][num_lb:].chunk(2)

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            # if distribution alignment hook is registered, call it
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            if self.calc_weight_method == 'mlp':
                mask = torch.sigmoid(example_logits_ulb_w)
            else:
                mask = torch.softmax(example_logits_ulb_w, 0) * example_logits_ulb_w.shape[0]
            select_loss = (1 - mask).mean() * self.l2_weight

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + select_loss

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/select_loss'] = select_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        return tb_dict