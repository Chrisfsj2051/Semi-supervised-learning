import random

import torch
import torch.nn.functional as F
import math
import numpy as np

from .hook import Hook


class DataDietHook(Hook):
    def __init__(self):
        super().__init__()
        self.idx2score = {}

    def compute_el2n_scores(self, confidences, labels):
        num_classes = confidences.shape[1]
        errors = confidences - F.one_hot(labels, num_classes)
        scores = torch.norm(errors, p=2, dim=1)
        return scores

    def compute_random_scores(self, confidences, labels, indices):
        indices = indices.cpu().tolist()
        scores = []
        for index in indices:
            if index not in self.idx2score.keys():
                self.idx2score[index] = np.random.randn()
            scores.append(self.idx2score[index])
        scores = confidences.new_tensor(scores)
        return scores

    def compute_scores(self, confidences, labels, indices, method):
        if method == 'el2n':
            return self.compute_el2n_scores(confidences, labels)
        elif method == 'random':
            return self.compute_random_scores(confidences, labels, indices)
        else:
            raise NotImplementedError('????')

    @torch.no_grad()
    def predict_nongrad_method(self, algorithm):
        # repeat doesn't matter
        confidence_list, label_list, idx_list, score_list = [], [], [], []
        model = algorithm.model
        training = model.module.training
        device = algorithm.model.device
        method = algorithm.args.datadiet_method
        model.eval()
        for data in algorithm.loader_dict['train_ulb']:
            idx, x = data['idx_ulb'].to(device), data['x_ulb_w'].to(device)
            outs = model(x)
            confidences = torch.softmax(outs['logits'], 1)
            _, labels = confidences.max(1)
            score_list.append(self.compute_scores(confidences, labels, idx, method))
            confidence_list.append(confidences)
            label_list.append(labels)
            idx_list.append(idx)

        model.train(training)
        confidences = torch.cat(confidence_list, 0)
        labels = torch.cat(label_list, 0)
        indices = torch.cat(idx_list, 0)
        scores = torch.cat(score_list, 0)
        confidence_scores, _ = confidences.max(1)
        threshold = algorithm.call_hook("get_threshold", "MaskingHook")
        mask = scores[labels] < threshold[labels]
        scores[mask] = -100 + confidence_scores[mask]
        results = {
            'confidences': confidences,
            'labels': labels,
            'indices': indices,
            'scores': scores
        }
        return results

    def predict_influence_method(self, algorithm):
        from semilearn.algorithms.utils.loss import ce_loss
        # 0. set init status
        model_params = algorithm.model.module.get_influence_function_params()
        device = algorithm.model.device
        lb_dset = algorithm.dataset_dict['train_lb']
        ulb_dset = algorithm.loader_dict['train_ulb']
        threshold = algorithm.call_hook("get_threshold", "MaskingHook")
        training = algorithm.model.training
        algorithm.model.eval()
        # 1. grad_val
        # TODO: EMA MC DROPOUT
        batch_size = algorithm.args.batch_size
        ulb_ratio = algorithm.args.uratio
        val_size = ulb_ratio * batch_size
        sampled_indices = np.random.choice(len(lb_dset), val_size * 2)
        sampled_indices = torch.from_numpy(sampled_indices)
        mixup_ratio = np.random.beta(1, 1, size=val_size)[:, None, None, None]
        mixup_ratio = torch.from_numpy(mixup_ratio).to(device).float()
        sampled_data = [lb_dset[i] for i in sampled_indices]
        sampled_x = mixup_ratio.new_tensor(np.concatenate([item['x_lb'][None] for item in sampled_data], 0))
        mixup_x = sampled_x[::2] * mixup_ratio + sampled_x[1::2] * (1 - mixup_ratio)
        sampled_y = mixup_ratio.new_tensor([item['y_lb'] for item in sampled_data]).long()
        mixup_y = sampled_y.new_zeros(mixup_x.shape[0], algorithm.num_classes).float()
        mixup_ratio = mixup_ratio.squeeze()[:, None]
        mixup_y.scatter_(1, sampled_y[::2, None], mixup_ratio)
        mixup_y.scatter_(1, sampled_y[1::2, None], 1 - mixup_ratio)
        val_logits = algorithm.model(mixup_x)['logits']
        val_loss = ce_loss(val_logits, mixup_y, reduction='mean')
        val_grads = torch.autograd.grad(val_loss, model_params)
        val_grads = torch.cat([x.flatten() for x in val_grads])
        # assert len(val_grads) == 1
        # val_grads = val_grads[0]

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

    def predict(self, algorithm):
        method = algorithm.args.datadiet_method
        if method == 'influence':
            return self.predict_influence_method(algorithm)
        else:
            return self.predict_nongrad_method(algorithm)

    def apply_prune(self, algorithm, num_keep, predictions):
        indices = predictions['indices']
        scores = predictions['scores']
        sorted_indices = torch.argsort(scores, descending=True)
        keep_indices = indices[sorted_indices[:num_keep]].cpu().tolist()
        random.shuffle(keep_indices)
        algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(keep_indices)

    def reset_status(self, algorithm):
        algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(None)
        self.idx2score = {}

    def before_train_step(self, algorithm):
        if algorithm.it > 1 and self.every_n_iters(algorithm, algorithm.args.datadiet_interval):
            algorithm.print_fn(f"Start pruning: epoch={algorithm.epoch}, it={algorithm.it}")
            self.reset_status(algorithm)
            args = algorithm.args
            total_samples = len(algorithm.loader_dict['train_ulb']) * args.batch_size * (1 + args.uratio)
            ratio = total_samples / len(algorithm.dataset_dict['train_ulb'])
            keep_num = math.ceil(args.datadiet_keep_num * ratio)
            predictions = self.predict(algorithm)
            self.apply_prune(algorithm, keep_num, predictions)
