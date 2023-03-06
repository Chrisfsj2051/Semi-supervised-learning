import torch
import math
import random
import numpy as np
from semilearn.data_diet.influence import DataDietInfluenceHook



class DataDietRetrieveHook(DataDietInfluenceHook):
    def __init__(self):
        super().__init__()
        self.id2weight = {}

    def get_batch_weight(self, algorithm, idx_ulb):
        try:
            # weights = [max(self.id2weight[int(i)], 0) for i in idx_ulb]
            weights = [max(self.id2weight[int(i)], 0) for i in idx_ulb]
            weights = idx_ulb.new_tensor(weights, dtype=torch.float).mean()
        except Exception:
            weights = [1.0 for _ in idx_ulb]
        return idx_ulb.new_tensor(weights, dtype=torch.float)

    def compute_val_gradient(self, algorithm):
        from semilearn.algorithms.utils.loss import ce_loss
        val_size = algorithm.args.batch_size * algorithm.args.uratio
        mixup_x, mixup_y = self.mixup_sampling(algorithm, algorithm.dataset_dict['train_lb'], val_size)
        model_output = algorithm.model(mixup_x)
        feat, logits = model_output['feat'], model_output['logits']
        embDim = feat.shape[1]
        loss = ce_loss(logits, mixup_y, reduction='none').sum()
        l0_grads = torch.autograd.grad(loss, logits)[0]
        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * feat.repeat(1, algorithm.num_classes)
        l0_grads = l0_grads.mean(dim=0).view(1, -1)
        l1_grads = l1_grads.mean(dim=0).view(1, -1)
        val_grads_per_elem = torch.cat((l0_grads.detach(), l1_grads.detach()), dim=1)
        sum_val_grad = torch.sum(val_grads_per_elem, dim=0)
        return {
            'sum_val_grad': sum_val_grad,
            'l1_grad': l1_grads,
            'l0_grad': l0_grads,
            'l1_out': feat.detach(),
            'l0_out': logits.detach()
        }

    def compute_example_gradient(self, algorithm):
        from semilearn.algorithms.utils.loss import ce_loss
        ulb_dset = algorithm.loader_dict['train_ulb']
        l0_grads_list, l1_grads_list, idx_list = [], [], []
        for ulb_data in ulb_dset:
            x_concat = torch.cat([ulb_data['x_ulb_w'], ulb_data['x_ulb_s']], 0)
            output_concat = algorithm.model(x_concat)
            logits_w, logits_s = output_concat['logits'].chunk(2)
            feat_w, feat_s = output_concat['feat'].chunk(2)
            conf_w = torch.softmax(logits_w, 1)
            pseudo_conf, pseudo_label = conf_w.max(1)
            ulb_loss = ce_loss(logits_s, pseudo_label, reduction='none').sum()
            embDim = feat_s.shape[1]
            l0_grads = torch.autograd.grad(ulb_loss, logits_s)[0]
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * feat_s.repeat(1, algorithm.num_classes)
            l0_grads = l0_grads.mean(dim=0).view(1, -1)
            l1_grads = l1_grads.mean(dim=0).view(1, -1)
            idx_list.append(ulb_data['idx_ulb'])
            l0_grads_list.append(l0_grads.detach())
            l1_grads_list.append(l1_grads.detach())

        l0_grads, l1_grads = torch.cat(l0_grads_list, 0), torch.cat(l1_grads_list, 0)
        return (torch.cat(idx_list), torch.cat([l0_grads, l1_grads], 1))

    def predict(self, algorithm):
        idxs, per_batch_grads = self.compute_example_gradient(algorithm)
        sum_val_grads = self.compute_val_gradient(algorithm)['sum_val_grad']
        return {
            'indices': idxs,
            'per_batch_grads': per_batch_grads,
            'sum_val_grads': sum_val_grads
        }

    def apply_prune(self, algorithm, num_keep, predictions):
        idxs, gammas = [], []
        per_batch_grads = predictions['per_batch_grads']
        sum_val_grads = predictions['sum_val_grads']
        idxs_temp, gammas_temp = self.ompwrapper(
            per_batch_grads.transpose(0, 1),
            sum_val_grads,
            math.ceil(num_keep / algorithm.loader_dict['train_ulb'].batch_size),
            device=per_batch_grads.device
        )
        batch_wise_indices = list(algorithm.loader_dict['train_ulb'].batch_sampler)
        for i in range(len(idxs_temp)):
            tmp = batch_wise_indices[idxs_temp[i]]
            idxs.extend(tmp)
            gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))

        diff = num_keep - len(idxs)
        if diff < 0:
            idxs, gammas = idxs[:diff], gammas[:diff]
        else:
            # remain_list = set(np.arange(len(algorithm.dataset_dict['train_ulb'])))
            # remain_list = list(remain_list.difference(set(idxs)))
            # random.shuffle(remain_list)
            remain_list = np.arange(len(algorithm.dataset_dict['train_ulb']))
            idxs.extend(remain_list[:diff])
            gammas.extend([1 for _ in range(diff)])

        for idx, gamma in zip(idxs, gammas):
            self.id2weight[idx] = gamma
        algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(idxs)
