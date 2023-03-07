import torch
import math
import random
import numpy as np
from semilearn.data_diet.gradmatch import DataDietGradMatchHook



class DataDietRetrieveHook(DataDietGradMatchHook       ):
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

    def eval_taylor_modular(self, grads):
        grads_val = self.grads_val_curr
        with torch.no_grad():
            gains = torch.matmul(grads, grads_val)
        return gains

    def _update_gradients_subset(self, grads_X, element):
        grads_X += self.grads_per_elem[element].sum(dim=0)

    def predict(self, algorithm):
        val_grad_map = self.compute_val_gradient(algorithm)
        l0_grads, l1_grads = val_grad_map['l0_grads'], val_grad_map['l1_grads']
        concat_grads = torch.cat((l0_grads.detach(), l1_grads.detach()), dim=1)
        grads_val_curr = torch.mean(concat_grads, dim=0)
        init_out, init_l1 = val_grad_map['l0_out'], val_grad_map['l1_out']
        per_example_grad_map = self.compute_example_gradient(algorithm)

        print('in')

    def greedy(self, budget):
        greedySet = list()
        N = self.grads_per_elem.shape[0]
        remainSet = list(range(N))
        # t_ng_start = time.time()  # naive greedy start time
        numSelected = 0
        subset_size = int((len(self.grads_per_elem) / budget) * math.log(100))
        while (numSelected < budget):
            # Try Using a List comprehension here!
            subset_selected = random.sample(remainSet, k=subset_size)
            rem_grads = self.grads_per_elem[subset_selected]
            gains = self.eval_taylor_modular(rem_grads)
            # Update the greedy set and remaining set
            _, indices = torch.sort(gains.view(-1), descending=True)
            bestId = [subset_selected[indices[0].item()]]
            greedySet.append(bestId[0])
            remainSet.remove(bestId[0])
            numSelected += 1
            # Update debug in grads_currX using element=bestId
            if numSelected > 1:
                self._update_gradients_subset(grads_curr, bestId)
            else:  # If 1st selection, then just set it to bestId grads
                grads_curr = self.grads_per_elem[bestId].view(1, -1)  # Making it a list so that is mutable!
            # Update the grads_val_current using current greedySet grads
            self._update_grads_val(grads_curr)
        # self.logger.debug("RETRIEVE's Stochastic Greedy selection time: %f", time.time() - t_ng_start)

        return list(greedySet), [1] * budget

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
