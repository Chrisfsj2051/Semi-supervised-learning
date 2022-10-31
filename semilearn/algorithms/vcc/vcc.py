# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.flexmatch import FlexMatch
import torch.nn.functional as F
import numpy as np
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool


class VCC(FlexMatch):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.vcc_training_warmup = args.vcc_training_warmup
        self.vcc_selection_warmup = args.vcc_selection_warmup
        self.vcc_unlab_loss_weight = args.vcc_unlab_loss_weight
        self.vcc_lab_loss_weight = args.vcc_lab_loss_weight
        self.only_supervised = args.vcc_only_supervised

    def compute_vcc_loss(self, recon_pred, recon_gt, logvar, mu, mask):
        recon_r_ulb_w_log_softmax = torch.log_softmax(recon_pred, -1)
        recon_loss = (torch.mean(-recon_gt * recon_r_ulb_w_log_softmax, 1) * mask).mean()
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1) * mask).mean()
        return {
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl_loss.mean()
        }

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                if self.only_supervised:
                    inputs = torch.cat((x_lb, x_lb, x_lb))
                else:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                recon_pred_ulb_w, recon_pred_ulb_s = outputs['recon_pred'][num_lb:].chunk(2)
                recon_gt_ulb_w, recon_gt_ulb_s = outputs['recon_gt'][num_lb:].chunk(2)
                mu_ulb_w, mu_ulb_s = outputs['mu'][num_lb:].chunk(2)
                logvar_ulb_w, logvar_ulb_s = outputs['logvar'][num_lb:].chunk(2)
                recon_pred_lb, recon_gt_lb = outputs['recon_pred'][:num_lb], outputs['recon_gt'][:num_lb]
                mu_lb, logvar_lb = outputs['mu'][:num_lb], outputs['logvar'][:num_lb]
                calibrated_logits_ulb_w, calibrated_logits_ulb_s = outputs['calibrated_logits'][num_lb:].chunk(2)
            else:
                raise NotImplementedError("Not implemented!")

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            vcc_logits = logits_x_ulb_w
            vcc_unlab_loss_weight = 0.0
            vcc_lab_loss_weight = 0.0

            if self.it > self.vcc_training_warmup:
                vcc_unlab_loss_weight = self.vcc_unlab_loss_weight
                vcc_lab_loss_weight = self.vcc_lab_loss_weight
                self.p_cutoff = self.args.p_cutoff
            if self.it > self.vcc_selection_warmup:
                vcc_logits = calibrated_logits_ulb_w
                self.p_cutoff = self.args.vcc_p_cutoff

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=vcc_logits, idx_ulb=idx_ulb)

            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=vcc_logits,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)

            unsup_loss = self.lambda_u * consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            vcc_loss_ulb_w = self.compute_vcc_loss(recon_pred_ulb_w, recon_gt_ulb_w, logvar_ulb_w, mu_ulb_w, mask)
            recon_loss_ulb_w = vcc_loss_ulb_w['recon_loss'] * vcc_unlab_loss_weight
            kl_loss_ulb_w = vcc_loss_ulb_w['kl_loss'] * vcc_unlab_loss_weight
            vcc_loss_lb = self.compute_vcc_loss(recon_pred_lb, recon_gt_lb, logvar_lb, mu_lb, torch.ones_like(mask))
            recon_loss_lb = vcc_loss_lb['recon_loss'] * vcc_lab_loss_weight
            kl_loss_lb = vcc_loss_lb['kl_loss'] * vcc_lab_loss_weight
            total_loss = (sup_loss + unsup_loss + recon_loss_ulb_w +
                          kl_loss_ulb_w + kl_loss_lb + recon_loss_lb)

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

    def predict(self, eval_dest):
        self.model.eval()
        self.ema.apply_shadow()
        eval_loader = self.loader_dict[eval_dest]
        total_loss, total_num = 0.0, 0.0
        y_true, y_pred, y_probs, y_logits = [], [], [], []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                output = self.model(x)
                logits = output['logits']
                calibrated_logits = output['calibrated_logits']

                loss = F.cross_entropy(logits, y, reduction='mean')
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.append(torch.softmax(calibrated_logits, dim=-1).cpu().numpy())
                total_loss += loss.item() * num_batch

        self.ema.restore()
        self.model.train()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        y_probs = np.concatenate(y_probs)
        return y_true, y_pred, y_logits, y_probs, total_loss, total_num
