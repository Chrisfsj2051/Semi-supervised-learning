from matplotlib import pyplot as plt
import torch

from semilearn.core.utils import send_model_cuda
from semilearn.datasets.cv_datasets.cifar import CIFAR10_CLASSES
from tools.utils import build_model

import torch.distributed as dist


def init_algorithm():
    algorithm = build_model()
    args = algorithm.args
    args.distributed = True
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    algorithm.model = send_model_cuda(algorithm.args, algorithm.model)
    algorithm.ema_model = send_model_cuda(algorithm.args, algorithm.ema_model)
    try:
        algorithm.load_model(algorithm.args.visualize_load_path)
    except Exception:
        print('Load model dict failed!!!!!!')
    algorithm.call_hook("before_run")
    algorithm.dataset_dict['eval'].alg = 'evaluation'
    return algorithm


@torch.no_grad()
def main():
    algorithm = init_algorithm()
    (y_true, y_pred, y_logits, y_probs, total_loss, total_num
     ) = algorithm.predict('eval')
    x = algorithm.dataset_dict['eval'].data
    for i in range(x.shape[0]):
        plt.imshow(x[i])
        plt.axis('off')
        fontsize = 16
        plt.text(0, 34, f'ground-truth: {CIFAR10_CLASSES[y_true[i]]}', fontdict={'fontsize': fontsize, 'color': 'r'})
        plt.text(0, 36, f'prediction: {CIFAR10_CLASSES[y_pred[i]]}', fontdict={'fontsize': fontsize, 'color': 'b'})
        plt.text(0, 38, f'confidence: {y_probs[i].max(): .2f}', fontdict={'fontsize': fontsize, 'color': 'b'})
        plt.tight_layout(pad=0.1)
        plt.show()

        # plt.savefig('tmp.png', bbox_inches='tight', pad_inches=0)
        # break


if __name__ == '__main__':
    main()
