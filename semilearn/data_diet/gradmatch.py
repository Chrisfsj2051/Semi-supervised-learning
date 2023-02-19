import torch

from semilearn.data_diet.influence import DataDietInfluenceHook


class DataDietGradMatchHook(DataDietInfluenceHook):
    def __init__(self):
        super().__init__()
        self.batchid2weight = {}

    # def get_batch_weight(self, algorithm, idx_ulb):
    #     raise NotImplementedError()

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

    def predict(self, algorithm):
        self.compute_val_gradient(algorithm)
        # lb_dset = algorithm.dataset_dict['train_lb']
        # ulb_dset = algorithm.loader_dict['train_ulb']


