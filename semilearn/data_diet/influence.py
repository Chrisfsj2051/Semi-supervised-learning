import torch
import numpy as np

from .base import DataDietBaseHook


class DataDietInfluenceHook(DataDietBaseHook):
    def __init__(self):
        super().__init__()
        self.idx2score = {}
        self.batchid2weight = {}

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
        return mixup_x, mixup_y

    def predict(self, algorithm):
        from semilearn.algorithms.utils.loss import ce_loss
        # 0. set init status
        model_params = algorithm.model.module.get_influence_function_params()
        lb_dset = algorithm.dataset_dict['train_lb']
        ulb_dset = algorithm.loader_dict['train_ulb']
        threshold = algorithm.call_hook("get_threshold", "MaskingHook")
        training = algorithm.model.training
        algorithm.model.eval()
        # 1. grad_val
        # TODO: EMA MC DROPOUT
        val_size = algorithm.args.batch_size * algorithm.args.uratio
        mixup_x, mixup_y = self.mixup_sampling(algorithm, lb_dset, val_size)
        val_logits = algorithm.model(mixup_x)['logits']
        val_loss = ce_loss(val_logits, mixup_y, reduction='mean')
        val_grads = torch.autograd.grad(val_loss, model_params)
        val_grads = torch.cat([x.flatten() for x in val_grads])

        # 2. Hessian (Identity)

        # 3. per example grad
        idx_list, scores_list = [], []
        for ulb_data in ulb_dset:
            idx = ulb_data['idx_ulb']
            x_concat = torch.cat([ulb_data['x_ulb_w'], ulb_data['x_ulb_s']], 0)
            output_concat = algorithm.model(x_concat)
            logits_w, logits_s = output_concat['logits'].chunk(2)
            conf_w = torch.softmax(logits_w, 1)
            pseudo_conf, pseudo_label = conf_w.max(1)
            # 1: selected; 0: ignored
            pseudo_mask = pseudo_conf > threshold[pseudo_label]
            # LRU????????
            ulb_loss = ce_loss(logits_s, pseudo_label, reduction='none')

            group_num = 16
            for (idx_group, loss_group, mask_group) in zip(
                    idx.chunk(group_num), ulb_loss.chunk(group_num),
                    pseudo_mask.chunk(group_num)):
                score_group = torch.full(mask_group.shape, -1e5)
                if mask_group.sum() > 0:
                    loss_group = loss_group[mask_group].mean()
                    example_grads = torch.autograd.grad(loss_group, model_params, retain_graph=True)
                    example_grads = torch.cat([x.flatten() for x in example_grads])
                    score_group[mask_group] = -(example_grads * val_grads).mean()
                idx_list.append(idx_group)
                scores_list.append(score_group)

        algorithm.model.train(mode=training)
        results = {
            'indices': torch.cat(idx_list),
            'scores': torch.cat(scores_list)
        }
        return results
