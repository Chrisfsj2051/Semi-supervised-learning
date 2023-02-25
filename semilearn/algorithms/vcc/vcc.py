# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.flexmatch import FlexMatch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from semilearn.algorithms.utils import ce_loss, consistency_loss


class VCC(FlexMatch):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.vcc_training_warmup = args.vcc_training_warmup
        self.vcc_selection_warmup = args.vcc_selection_warmup
        self.vcc_unlab_recon_loss_weight = args.vcc_unlab_recon_loss_weight
        self.vcc_unlab_kl_loss_weight = args.vcc_unlab_kl_loss_weight
        self.vcc_lab_loss_weight = args.vcc_lab_loss_weight
        self.only_supervised = args.vcc_only_supervised
        num_ulb = len(self.dataset_dict['train_ulb'])
        self.uncertainty_selected = torch.zeros(num_ulb)
        self.uncertainty_ema_map = torch.zeros(num_ulb, args.num_classes)
        self.uncertainty_ema_step = args.vcc_mc_upd_ratio

    def compute_vcc_loss(self, recon_pred, recon_gt, logvar, mu, mask):
        if self.args.vcc_recon_loss == 'cross_entropy':
            recon_r_log_softmax = torch.log_softmax(recon_pred, -1)
            recon_loss = (torch.mean(-recon_gt * recon_r_log_softmax, 1) * mask).mean()
        elif self.args.vcc_recon_loss == 'mae' or self.args.vcc_recon_loss == 'mse':
            recon_r_softmax = torch.softmax(recon_pred, -1)
            if self.args.vcc_recon_loss == 'mae':
                recon_loss = torch.nn.L1Loss()(recon_r_softmax, recon_gt)
            else:
                recon_loss = torch.nn.MSELoss()(recon_r_softmax, recon_gt)

        kl_loss = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1) * mask).mean()
        return {
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl_loss.mean()
        }

    def update_uncertainty_map(self, idx_ulb, recon_gt_ulb_w):
        if dist.get_world_size() > 1:
            dist_idx_ulb = idx_ulb.new_zeros(self.uncertainty_selected.shape[0])
            dist_upd_val = recon_gt_ulb_w.new_zeros(self.uncertainty_selected.shape[0], recon_gt_ulb_w.shape[1])
            dist_idx_ulb[idx_ulb], dist_upd_val[idx_ulb] = 1, recon_gt_ulb_w
            dist.all_reduce(dist_idx_ulb, op=dist.ReduceOp.SUM)
            dist.all_reduce(dist_upd_val, op=dist.ReduceOp.SUM)
            dist.barrier()
            dist_upd_val = dist_upd_val / (dist_idx_ulb[..., None] + 1e-7)
            recon_gt_ulb_w = dist_upd_val[idx_ulb]
            dist.barrier()

        self.uncertainty_ema_map = self.uncertainty_ema_map.to(self.gpu)
        update_weight = torch.ones_like(recon_gt_ulb_w)
        update_weight[self.uncertainty_selected[idx_ulb] == 1] = self.uncertainty_ema_step
        self.uncertainty_selected[idx_ulb] = 1
        updated_value = update_weight * recon_gt_ulb_w + (1 - update_weight) * self.uncertainty_ema_map[idx_ulb].cuda()
        self.uncertainty_ema_map[idx_ulb] = updated_value
        return updated_value

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
            vcc_unlab_recon_loss_weight, vcc_unlab_kl_loss_weight = 0.0, 0.0
            vcc_lab_loss_weight = 0.0
            softmax_x_ulb = True

            if self.it > self.vcc_training_warmup:
                loss_warmup_alpha = min((self.it - self.vcc_training_warmup) / 100, 1.0)
                vcc_unlab_recon_loss_weight = self.vcc_unlab_recon_loss_weight * loss_warmup_alpha
                vcc_unlab_kl_loss_weight = self.vcc_unlab_kl_loss_weight * loss_warmup_alpha
                vcc_lab_loss_weight = self.vcc_lab_loss_weight * loss_warmup_alpha
                self.p_cutoff = self.args.p_cutoff
            if self.it > self.vcc_selection_warmup:
                if self.args.vcc_disable_variance:
                    vcc_logits = recon_gt_ulb_w
                    softmax_x_ulb = False
                else:
                    vcc_logits = calibrated_logits_ulb_w
                self.p_cutoff = self.args.vcc_p_cutoff

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=vcc_logits, idx_ulb=idx_ulb,
                                  softmax_x_ulb=softmax_x_ulb)

            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=vcc_logits,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)

            unsup_loss = self.lambda_u * consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            recon_gt_ulb_w = self.update_uncertainty_map(idx_ulb, recon_gt_ulb_w)
            vcc_loss_ulb_w = self.compute_vcc_loss(recon_pred_ulb_w, recon_gt_ulb_w, logvar_ulb_w, mu_ulb_w, mask)
            recon_loss_ulb_w = vcc_loss_ulb_w['recon_loss'] * vcc_unlab_recon_loss_weight
            kl_loss_ulb_w = vcc_loss_ulb_w['kl_loss'] * vcc_unlab_kl_loss_weight
            vcc_loss_lb = self.compute_vcc_loss(recon_pred_lb, recon_gt_lb, logvar_lb, mu_lb,
                                                mask.new_ones(recon_pred_lb.shape[0]))
            recon_loss_lb = vcc_loss_lb['recon_loss'] * vcc_lab_loss_weight
            kl_loss_lb = vcc_loss_lb['kl_loss'] * vcc_lab_loss_weight
            total_loss = (sup_loss + unsup_loss + recon_loss_ulb_w +
                          kl_loss_ulb_w + kl_loss_lb + recon_loss_lb)
            # total_loss *= 0
            # for index in range(10):
            #     print('vae: \n', calibrated_logits_ulb_w.softmax(1)[index].topk(1))
            #     print('gt: \n', recon_gt_ulb_w[index].topk(1), '\n...........')

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
        save_dict['uncertainty_selected'] = self.uncertainty_selected.cpu()
        save_dict['uncertainty_ema_map'] = self.uncertainty_ema_map.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.uncertainty_selected = checkpoint['uncertainty_selected'].cuda(self.gpu)
        self.uncertainty_ema_map = checkpoint['uncertainty_ema_map'].cuda(self.gpu)
        self.print_fn("additional VCC parameter loaded")
        return checkpoint

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

                loss = F.cross_entropy(logits, y, reduction='mean')
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                if self.args.vcc_disable_variance:
                    y_probs.append(output['recon_gt'].cpu().numpy())
                else:
                    y_probs.append(torch.softmax(output['calibrated_logits'], dim=-1).cpu().numpy())
                total_loss += loss.item() * num_batch

        self.ema.restore()
        self.model.train()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        y_probs = np.concatenate(y_probs)
        return y_true, y_pred, y_logits, y_probs, total_loss, total_num
