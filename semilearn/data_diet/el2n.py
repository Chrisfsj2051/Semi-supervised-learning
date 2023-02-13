from .random import DataDietRandomHook
import torch.nn.functional as F
import torch


class DataDietEL2NHook(DataDietRandomHook):

    def compute_random_scores(self, confidences, labels, indices):
        num_classes = confidences.shape[1]
        errors = confidences - F.one_hot(labels, num_classes)
        scores = torch.norm(errors, p=2, dim=1)
        return scores
