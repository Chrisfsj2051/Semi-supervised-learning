import torch
import math
import random
import numpy as np
from semilearn.data_diet.gradmatch import DataDietGradMatchHook


class DataDietRetrieveHook(DataDietGradMatchHook):
    def __init__(self):
        super().__init__()
        self.eta = 0.03

    def get_batch_weight(self, algorithm, idx_ulb):
        return 1.0

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            gains = torch.matmul(grads, grads_val)
        return gains

    def _update_gradients_subset(self, grads_X, element):
        grads_X += self.grads_per_elem[element].sum(dim=0)

    def _update_grads_val(self, algorithm, grads_currX=None, first_init=False):
        from semilearn.algorithms.utils.loss import ce_loss
        if first_init:
            val_grad_map = self.compute_val_gradient(algorithm)
            l0_grads, l1_grads = val_grad_map['l0_grads'], val_grad_map['l1_grads']
            concat_grads = torch.cat((l0_grads.detach(), l1_grads.detach()), dim=1)
            self.grads_val_curr = torch.mean(concat_grads, dim=0).view(-1, 1)
            self.init_out = val_grad_map['l0_out_with_graph']
            self.init_l1 = val_grad_map['l1_out_with_graph']
            self.y_val = val_grad_map['target']
        elif grads_currX is not None:
            out_vec = self.init_out - (
                    self.eta *
                    grads_currX[0][0:algorithm.num_classes].view(1, -1).expand(self.init_out.shape[0], -1))
            out_vec = out_vec - (
                    self.eta *
                    torch.matmul(
                        self.init_l1,
                        grads_currX[0][algorithm.num_classes:].view(algorithm.num_classes, -1).transpose(0, 1)
                    )
            )
            loss = ce_loss(out_vec, self.y_val).sum()
            l0_grads = torch.autograd.grad(loss, out_vec)[0]
            l0_expand = torch.repeat_interleave(l0_grads, self.init_l1.shape[1], dim=1)
            l1_grads = l0_expand * self.init_l1.repeat(1, algorithm.num_classes)
            self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0).view(-1, 1)

    def predict(self, algorithm):
        self.grads_per_elem = self.compute_example_gradient(algorithm)['per_batch_grads']
        self._update_grads_val(algorithm, first_init=True)
        return {}

    def greedy(self, algorithm, budget):
        budget_batch = math.floor(budget / algorithm.loader_dict['train_ulb'].batch_size)
        greedySet = list()
        N = self.grads_per_elem.shape[0]
        remainSet = list(range(N))
        # t_ng_start = time.time()  # naive greedy start time
        numSelected = 0
        subset_size = int((len(self.grads_per_elem) / budget_batch) * math.log(100))
        subset_size = max(subset_size, 1)
        while (numSelected < budget_batch):
            # Try Using a List comprehension here!
            subset_selected = random.sample(remainSet, k=subset_size)
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            _, indices = torch.sort(gains.view(-1), descending=True)
            # if len(indices) == 0:
            #     print(f'indices={indices}, remainSetSize={len(remainSet)},'
            #           f'subsetSize={subset_size}, lenGardNorm={len(self.grads_per_elem)}')
            bestId = [subset_selected[indices[0].item()]]
            greedySet.append(bestId[0])
            remainSet.remove(bestId[0])
            numSelected += 1
            # Update debug in grads_currX using element=bestId
            if numSelected > 1:
                self._update_gradients_subset(grads_curr, bestId)
            else:
                grads_curr = self.grads_per_elem[bestId].view(1, -1)
            self._update_grads_val(algorithm, grads_curr)
        # self.logger.debug("RETRIEVE's Stochastic Greedy selection time: %f", time.time() - t_ng_start)

        return list(greedySet)

    def apply_prune(self, algorithm, num_keep, predictions):
        idxs = []
        idxs_temp = self.greedy(algorithm, num_keep)
        batch_wise_indices = list(algorithm.loader_dict['train_ulb'].batch_sampler)
        for i in range(len(idxs_temp)):
            tmp = batch_wise_indices[idxs_temp[i]]
            idxs.extend(tmp)
        algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(idxs)
