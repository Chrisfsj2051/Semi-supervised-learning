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
        confidences = confidences.max(1)[0]
        low_confidence_mask = confidences < 0.5
        scores[low_confidence_mask] = -1 + confidences[low_confidence_mask]
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
    def predict(self, algorithm):
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
        results = {
            'confidences': confidences,
            'labels': labels,
            'indices': indices,
            'scores': scores
        }
        return results

    def apply_prune(self, algorithm, num_keep, predictions):
        indices = predictions['indices']
        scores = predictions['scores']
        sorted_indices = torch.argsort(scores, descending=True)
        keep_indices = indices[sorted_indices[:num_keep]].cpu().tolist()
        algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(keep_indices)

    def reset_status(self, algorithm):
        algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(None)
        self.idx2score = {}

    def before_train_epoch(self, algorithm):
        if algorithm.epoch > 10 and self.every_n_epochs(algorithm, algorithm.args.datadiet_interval):
            self.reset_status(algorithm)
            total_num = len(algorithm.loader_dict['train_ulb'])
            keep_num = (math.ceil((1 - algorithm.args.datadiet_drop_ratio) * total_num) *
                        algorithm.loader_dict['train_ulb'].batch_size)
            predictions = self.predict(algorithm)
            self.apply_prune(algorithm, keep_num, predictions)
