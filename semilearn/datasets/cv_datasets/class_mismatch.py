import random
import numpy as np
from .cifar import get_cifar


def get_class_mismatch_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    orig_num_classes = num_classes
    if name == 'class_mismatch_cifar10':
        orig_num_classes = 10
        orig_name = 'cifar10'
    elif name == 'class_mismatch_cifar100':
        orig_num_classes = 100
        orig_name = 'cifar100'
    else:
        raise NotImplementedError("Oh dude!")
    lb_dset, ulb_dset, eval_dset = get_cifar(args, alg, orig_name, num_labels, orig_num_classes,
                                             data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)

    lb_mask = lb_dset.targets < args.num_classes
    lb_dset.data = lb_dset.data[lb_mask]
    lb_dset.targets = lb_dset.targets[lb_mask]

    eval_mask = np.array(eval_dset.targets) < args.num_classes
    eval_dset.data = eval_dset.data[eval_mask]
    eval_dset.targets = np.array(eval_dset.targets)[eval_mask].tolist()

    ulb_data, ulb_targets = ulb_dset.data, ulb_dset.targets
    ulb_mask = ulb_targets < args.num_classes
    total_num = ulb_mask.sum()
    dirty_num = int(args.class_mismatch_ratio * total_num)
    clean_num = total_num - dirty_num

    def shuffle_and_sample(data, targets, sample_num):
        assert len(data) == len(targets)
        index = list(range(0, len(data)))
        random.seed(args.seed)
        index = np.array(random.sample(index, sample_num))
        return data[index], targets[index]

    clean_data, clean_targets = shuffle_and_sample(ulb_data[ulb_mask], ulb_targets[ulb_mask], clean_num)
    dirty_data, dirty_targets = shuffle_and_sample(ulb_data[~ulb_mask], ulb_targets[~ulb_mask], dirty_num)
    ulb_dset.data = np.concatenate([clean_data, dirty_data], 0)
    ulb_dset.targets = np.concatenate([clean_targets, dirty_targets], 0)

    return lb_dset, ulb_dset, eval_dset