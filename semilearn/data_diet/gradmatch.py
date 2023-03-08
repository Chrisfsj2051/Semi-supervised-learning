import torch
import math
import random
import numpy as np
from semilearn.data_diet.influence import DataDietInfluenceHook


# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_Parallel_V1(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      nnz = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []

    argmin = torch.tensor([-1])
    for i in range(nnz):
        if resid.norm().item() / normb < tol:
            break
        projections = torch.matmul(AT, resid)  # AT.dot(resid)
        # print("Projections",projections.shape)

        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))

        if index not in indices:
            indices.append(index)

        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / torch.dot(A_i, A_i).view(-1)  # A_i.T.dot(A_i)
            A_i = A[:, index].view(1, -1)
        else:
            # print(indices)
            A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
            # print(x_i.shape)
            if positive:
                while min(x_i) < 0.0:
                    # print("Negative",b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape)
                    argmin = torch.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                    dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)  # A_i.T.dot(x_i)
    x_i = x_i.view(-1)
    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return x


class DataDietGradMatchHook(DataDietInfluenceHook):
    def __init__(self):
        super().__init__()
        self.id2weight = {}
        self.lam = 0.5
        self.eps = 1e-100

    def get_batch_weight(self, algorithm, idx_ulb):
        try:
            weights = [max(self.id2weight[int(i)], 0) for i in idx_ulb]
            # weights = [max(self.id2weight[int(i)], 0) for i in idx_ulb]
            # weights = idx_ulb.new_tensor(weights, dtype=torch.float).mean()
        except Exception:
            weights = [1.0 for _ in idx_ulb]
        return idx_ulb.new_tensor(weights, dtype=torch.float)

    def ompwrapper(self, X, Y, bud, device):
        reg = OrthogonalMP_REG_Parallel_V1(X, Y, nnz=bud,
                                           positive=True, lam=self.lam,
                                           tol=self.eps, device=device)
        ind = torch.nonzero(reg).view(-1)
        return ind.tolist(), reg[ind].tolist()

    def compute_val_gradient(self, algorithm):
        from semilearn.algorithms.utils.loss import ce_loss
        algorithm.model.zero_grad()
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
        algorithm.model.zero_grad()
        return {
            'sum_val_grad': sum_val_grad,
            'l1_grads': l1_grads,
            'l0_grads': l0_grads,
            'l1_out_with_graph': feat,
            'l0_out_with_graph': logits,
            'target': mixup_y.detach()
        }

    def compute_example_gradient(self, algorithm):
        from semilearn.algorithms.utils.loss import ce_loss
        algorithm.model.zero_grad()
        ulb_loader = algorithm.loader_dict['train_ulb']
        l0_grads_list, l1_grads_list, idx_list = [], [], []
        mask_list, pseudo_label_list = [], []
        for ulb_data in ulb_loader:
            x_concat = torch.cat([ulb_data['x_ulb_w'], ulb_data['x_ulb_s']], 0)
            output_concat = algorithm.model(x_concat)
            logits_w, logits_s = output_concat['logits'].chunk(2)
            feat_w, feat_s = output_concat['feat'].chunk(2)
            conf_w = torch.softmax(logits_w, 1)
            pseudo_conf, pseudo_label = conf_w.max(1)
            mask = algorithm.call_hook("masking", "MaskingHook", logits_x_ulb=logits_w, idx_ulb=None)
            ulb_loss = (ce_loss(logits_s, pseudo_label, reduction='none') * mask).sum()
            embDim = feat_s.shape[1]
            l0_grads = torch.autograd.grad(ulb_loss, logits_s)[0]
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * feat_s.repeat(1, algorithm.num_classes)
            l0_grads = l0_grads.mean(dim=0).view(1, -1)
            l1_grads = l1_grads.mean(dim=0).view(1, -1)
            idx_list.append(ulb_data['idx_ulb'])
            l0_grads_list.append(l0_grads.detach())
            mask_list.append(mask.detach())
            pseudo_label_list.append(pseudo_label.detach())
            l1_grads_list.append(l1_grads.detach())

        l0_grads, l1_grads = torch.cat(l0_grads_list, 0), torch.cat(l1_grads_list, 0)
        algorithm.model.zero_grad()
        return {
            'indices': torch.cat(idx_list),
            'per_batch_grads': torch.cat([l0_grads, l1_grads], 1),
            'pseudo_labels': torch.cat(pseudo_label_list, 0),
            'masks': torch.cat(mask_list, 0)
        }

    def predict(self, algorithm):
        per_example_grad_map = self.compute_example_gradient(algorithm)
        sum_val_grads = self.compute_val_gradient(algorithm)['sum_val_grad']
        return {
            'indices': per_example_grad_map['indices'],
            'per_batch_grads': per_example_grad_map['per_batch_grads'],
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
            remain_list = set(np.arange(len(algorithm.dataset_dict['train_ulb'])))
            remain_list = list(remain_list.difference(set(idxs)))
            random.shuffle(remain_list)
            # remain_list = np.arange(len(algorithm.dataset_dict['train_ulb']))
            idxs.extend(remain_list[:diff])
            gammas.extend([1 for _ in range(diff)])

        for idx, gamma in zip(idxs, gammas):
            self.id2weight[idx] = gamma
        algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(idxs)
