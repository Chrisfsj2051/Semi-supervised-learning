import random

import torch
import math

from semilearn.core.hooks import Hook


class DataDietBaseHook(Hook):
    def __init__(self):
        super().__init__()
        self.idx2score = {}

    def predict(self, algorithm):
        raise NotImplementedError()

    def get_batch_weight(self, algorithm, idx_ulb):
        return 1.0

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

    def process(self, algorithm):
        algorithm.print_fn(f"Start pruning: method={str(self.__class__)[:-2].split('.')[-1]}, "
                           f"epoch={algorithm.epoch}, it={algorithm.it}")
        self.reset_status(algorithm)
        args = algorithm.args
        ulb_loader = algorithm.loader_dict['train_ulb']
        ratio = args.datadiet_keep_num / len(algorithm.dataset_dict['train_ulb'])
        keep_num = math.ceil(len(ulb_loader) * ratio) * ulb_loader.batch_size
        predictions = self.predict(algorithm)
        self.apply_prune(algorithm, keep_num, predictions)

    def before_run(self, algorithm):
        if algorithm.epoch >= algorithm.args.datadiet_warmup_epoch:
            self.process(algorithm)

    def before_train_epoch(self, algorithm):
        apply_data_diet = algorithm.epoch > 1 and self.every_n_epochs(algorithm, algorithm.args.datadiet_interval)
        apply_data_diet |= algorithm.epoch == algorithm.args.datadiet_warmup_epoch
        if apply_data_diet:
            self.process(algorithm)
