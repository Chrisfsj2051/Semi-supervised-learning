import torch
import numpy as np

from ..nets.wrn.wrn_vcc import VariationalConfidenceCalibration

try:
    import functorch
except Exception:
    functorch = None
import random
import torch.distributed as dist
from .base import DataDietBaseHook


class DataDietInfluenceHook(DataDietBaseHook):
    def __init__(self):
        super().__init__()
        self.idx2score = {}
        self.batchid2weight = {}
        if functorch is None:
            raise ModuleNotFoundError('Please Install Functorch')

    def mixup_sampling(self, algorithm, dset, sample_size):
        sampled_indices = np.random.choice(len(dset), sample_size * 2)
        sampled_indices = torch.from_numpy(sampled_indices)
        mixup_ratio = np.random.beta(1, 1, size=sample_size)[:, None, None, None]
        mixup_ratio = torch.from_numpy(mixup_ratio).to(algorithm.model.device).float()
        sampled_data = [dset[i] for i in sampled_indices]
        sampled_x = mixup_ratio.new_tensor(np.concatenate([item['x_lb'][None] for item in sampled_data], 0))
        mixup_x = sampled_x[::2] * mixup_ratio + sampled_x[1::2] * (1 - mixup_ratio)
        sampled_y = mixup_ratio.new_tensor([item['y_lb'] for item in sampled_data]).long()
        mixup_y = sampled_y.new_zeros(mixup_x.shape[0], algorithm.num_classes).float()
        mixup_ratio = mixup_ratio.squeeze()[:, None]
        mixup_y.scatter_(1, sampled_y[::2, None], mixup_ratio)
        mixup_y.scatter_(1, sampled_y[1::2, None], 1 - mixup_ratio)
        return mixup_x, mixup_y, sampled_x, mixup_ratio

    def compute_val_grads(self, algorithm, model_params):
        from semilearn.algorithms.utils.loss import ce_loss
        val_size = algorithm.args.batch_size * algorithm.args.uratio
        lb_dset = algorithm.dataset_dict['train_lb']
        if algorithm.args.datadiet_val_grad_method == 'mixup':
            mixup_x, mixup_y, _, _ = self.mixup_sampling(algorithm, lb_dset, val_size)
            if isinstance(algorithm.model.module, VariationalConfidenceCalibration):
                val_logits = algorithm.model(algorithm, mixup_x)['logits']
            else:
                val_logits = algorithm.model(mixup_x)['logits']
        else:
            _, mixup_y, sampled_x, mixup_ratio = self.mixup_sampling(algorithm, lb_dset, val_size)
            val_feats = algorithm.model(sampled_x, only_feat=True)
            mixup_val_feats = val_feats[::2] * mixup_ratio + val_feats[1::2] * (1 - mixup_ratio)
            val_logits = algorithm.model.module.fc(mixup_val_feats)
        val_loss = ce_loss(val_logits, mixup_y, reduction='mean')
        val_grads = torch.autograd.grad(val_loss, model_params)
        val_grads = torch.cat([x.flatten() for x in val_grads])
        return val_grads

    def compute_example_score(self, algorithm, val_grads, model_params):
        from semilearn.algorithms.utils.loss import ce_loss
        ulb_loader = algorithm.loader_dict['train_ulb']
        idx_list, scores_list, pseudo_conf_list = [], [], []
        for ulb_data in ulb_loader:
            idx = ulb_data['idx_ulb']
            x_concat = torch.cat([ulb_data['x_ulb_w'], ulb_data['x_ulb_s']], 0)
            if isinstance(algorithm.model.module, VariationalConfidenceCalibration):
                output_concat = algorithm.model(algorithm, x_concat)
            else:
                output_concat = algorithm.model(x_concat)
            logits_w, logits_s = output_concat['logits'].chunk(2)
            conf_w = torch.softmax(logits_w, 1).detach()
            pseudo_conf, pseudo_label = conf_w.max(1)
            # 1: selected; 0: ignored
            pseudo_mask = algorithm.call_hook(
                "masking", "MaskingHook", logits_x_ulb=logits_w, idx_ulb=None).bool()
            ulb_loss = ce_loss(logits_s, pseudo_label, reduction='none') * pseudo_mask

            def single_example_grad(v):
                return torch.autograd.grad(ulb_loss, model_params, v)

            batch_score = (torch.randn(ulb_loss.shape[0]) - 100).to(ulb_loss.device)
            rank, world_size = dist.get_rank(), dist.get_world_size()
            group_size = algorithm.args.datadiet_influence_group_size // world_size
            group_pseudo_mask = pseudo_mask.view(-1, group_size)

            if algorithm.args.datadiet_exp_version == 1:
                ulb_loss = ce_loss(logits_s, pseudo_label, reduction='none').view(-1, group_size).sum(1)
            else:
                ulb_loss = ulb_loss.view(-1, group_size).sum(1) / (1e-5 + group_pseudo_mask.sum(1))
            I_N = torch.eye(ulb_loss.shape[0]).to(ulb_loss.device)
            batch_grads = functorch.vmap(single_example_grad)(I_N)
            batch_grads = [x.flatten(start_dim=1) for x in batch_grads]
            batch_grads = torch.cat(batch_grads, 1)
            batch_grads = batch_grads.repeat_interleave(group_size, 0)
            if algorithm.args.datadiet_exp_version == 1:
                batch_score = (batch_grads * torch.pow(pseudo_conf[..., None], 1.5)).mean(1).detach()
            else:
                batch_score[pseudo_mask] = -(batch_grads[pseudo_mask] * val_grads).mean(1).detach()
            idx_list.append(idx)
            scores_list.append(batch_score)
            pseudo_conf_list.append(pseudo_conf)

        return {
            'indices': torch.cat(idx_list),
            'scores': torch.cat(scores_list),
            'confidences': torch.cat(pseudo_conf_list)
        }

    def predict(self, algorithm):
        model_params = algorithm.model.module.get_influence_function_params(
            algorithm.args.datadiet_grad_params)
        training = algorithm.model.training
        algorithm.model.eval()

        val_grads = self.compute_val_grads(algorithm, model_params)
        results = self.compute_example_score(algorithm, val_grads, model_params)

        algorithm.model.train(mode=training)

        return {
            'indices': results['indices'],
            'scores': results['scores'],
            'confidences': results['confidences']
        }

    def apply_prune(self, algorithm, num_keep, predictions):
        if algorithm.args.datadiet_exp_version == 2:
            indices = predictions['indices']
            scores = predictions['scores']
            scores[scores < 0] = torch.rand(scores[scores < 0].shape).to(scores.device) * scores[scores >= 0].min()
            scores += 1e-10
            indices, scores = indices.cpu().numpy(), scores.cpu().numpy()
            scores = scores / scores.sum()
            keep_indices = np.random.choice(indices, size=num_keep, replace=False, p=scores)
            keep_indices = list(keep_indices)
            algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(keep_indices)
        else:
            return super(DataDietInfluenceHook, self).apply_prune(
                algorithm, num_keep, predictions)
