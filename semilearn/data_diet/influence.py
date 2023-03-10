import torch
import numpy as np
import functorch
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
        model_params = algorithm.model.module.get_influence_function_params(
            algorithm.args.datadiet_grad_params)
        lb_dset = algorithm.dataset_dict['train_lb']
        ulb_loader = algorithm.loader_dict['train_ulb']
        training = algorithm.model.training
        # algorithm.model.zero_grad()
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
        group_num = algorithm.args.datadiet_influence_group_size
        idx_list, scores_list = [], []
        # import time
        # tmp, tt = 0, time.time()
        for ulb_data in ulb_loader:
            # print(tmp, time.time() - tt)
            # tmp += 1
            # tt = time.time()
            # if tmp == 100:
            #     break
            idx = ulb_data['idx_ulb']
            x_concat = torch.cat([ulb_data['x_ulb_w'], ulb_data['x_ulb_s']], 0)
            output_concat = algorithm.model(x_concat)
            logits_w, logits_s = output_concat['logits'].chunk(2)
            conf_w = torch.softmax(logits_w, 1)
            pseudo_conf, pseudo_label = conf_w.max(1)
            # 1: selected; 0: ignored
            pseudo_mask = algorithm.call_hook(
                "masking", "MaskingHook", logits_x_ulb=logits_w, idx_ulb=None).bool()
            ulb_loss = ce_loss(logits_s, pseudo_label, reduction='none')

            def single_example_grad(v):
                return torch.autograd.grad(ulb_loss, model_params, v)

            batch_score = (torch.randn(ulb_loss.shape[0]) - 1e4).to(ulb_loss.device)
            I_N = torch.eye(ulb_loss.shape[0]).to(ulb_loss.device)
            batch_grads = functorch.vmap(single_example_grad)(I_N)
            batch_grads = [x.flatten(start_dim=1) for x in batch_grads]
            batch_grads = torch.cat(batch_grads, 1)
            batch_score[pseudo_mask] = -(batch_grads[pseudo_mask] * val_grads).mean(1).detach()
            idx_list.append(idx)
            scores_list.append(batch_score)

        # algorithm.model.zero_grad()
        algorithm.model.train(mode=training)
        results = {
            'indices': torch.cat(idx_list),
            'scores': torch.cat(scores_list)
        }
        return results
