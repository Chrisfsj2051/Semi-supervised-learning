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
        self.batchid2weight = {}
        self.lam = 0.5
        self.eps = 1e-100

    # def get_batch_weight(self, algorithm, idx_ulb):
    #     raise NotImplementedError()

    def ompwrapper(self, X, Y, bud, device):
        reg = OrthogonalMP_REG_Parallel_V1(X, Y, nnz=bud,
                                           positive=True, lam=self.lam,
                                           tol=self.eps, device=device)
        ind = torch.nonzero(reg).view(-1)
        return ind.tolist(), reg[ind].tolist()

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
        val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        sum_val_grad = torch.sum(val_grads_per_elem, dim=0)
        return sum_val_grad

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
            l0_grads_list.append(l0_grads)
            l1_grads_list.append(l1_grads)

        l0_grads, l1_grads = torch.cat(l0_grads_list, 0), torch.cat(l1_grads_list, 0)
        return (torch.cat(idx_list), torch.cat([l0_grads, l1_grads], 1))

    def predict(self, algorithm):
        idxs, per_batch_grads = self.compute_example_gradient(algorithm)
        sum_val_grads = self.compute_val_gradient(algorithm)
        return {
            'indices': idxs,
            'per_batch_grads': per_batch_grads,
            'sum_val_grads': sum_val_grads
        }

    def apply_prune(self, algorithm, num_keep, predictions):
        idxs, gammas = [], []
        ori_idxs = predictions['indices']
        per_batch_grads = predictions['per_batch_grads']
        sum_val_grads = predictions['sum_val_grads']
        idxs_temp, gammas_temp = self.ompwrapper(
            per_batch_grads.transpose(0, 1),
            sum_val_grads,
            math.ceil(num_keep / algorithm.loader_dict['train_ulb'].batch_size),
            device = per_batch_grads.device
        )
        batch_wise_indices = list(algorithm.loader_dict['train_ulb'].batch_sampler)
        for i in range(len(idxs_temp)):
            tmp = batch_wise_indices[idxs_temp[i]]
            idxs.extend(tmp)
            gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))

        diff = num_keep - len(idxs)
        if diff < 0:
            idxs, gammas = idxs[:diff], idxs[:diff]
        else:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))

        # sorted_indices = torch.argsort(scores, descending=True)
        # keep_indices = indices[sorted_indices[:num_keep]].cpu().tolist()
        # random.shuffle(keep_indices)
        # algorithm.loader_dict['train_ulb'].sampler.set_pruned_indices(keep_indices)

        # batch_wise_indices = list(self.trainloader.batch_sampler)
        # for i in range(len(idxs_temp)):
        #     tmp = batch_wise_indices[idxs_temp[i]]
        #     idxs.extend(tmp)
        #     gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))
        # lb_dset = algorithm.dataset_dict['train_lb']
        # ulb_dset = algorithm.loader_dict['train_ulb']
