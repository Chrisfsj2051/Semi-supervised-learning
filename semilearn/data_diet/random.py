import torch
import numpy as np

from .base import DataDietBaseHook


class DataDietRandomHook(DataDietBaseHook):
    def __init__(self):
        super().__init__()
        self.idx2score = {}
        self.batchid2weight = {}

    def compute_scores(self, confidences, labels, indices):
        indices = indices.cpu().tolist()
        scores = []
        for index in indices:
            if index not in self.idx2score.keys():
                self.idx2score[index] = np.random.randn()
            scores.append(self.idx2score[index])
        scores = confidences.new_tensor(scores)
        return scores

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
            score_list.append(self.compute_scores(confidences, labels, idx))
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
