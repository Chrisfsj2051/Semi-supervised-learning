import torch

from semilearn.core.utils.calibration_metrics import _populate_bins, BIN_ACC, expected_calibration_error
from tools.visualize_confidence import init_algorithm


@torch.no_grad()
def main():
    algorithm = init_algorithm()
    (y_true, y_pred, y_logits, y_probs, total_loss, total_num
     ) = algorithm.predict('eval')
    confs = y_probs.max(1)
    bin_dict = _populate_bins(confs, y_pred, y_true, 10)
    ece = expected_calibration_error(confs, y_pred, y_true, 10)
    import matplotlib.pyplot as plt
    bin_x = [f'0.{x}' for x in range(10)]
    bin_y = [bin_dict[i][BIN_ACC] for i in range(10)]
    rects = plt.bar(bin_x, bin_y, width=0.8, align='center')
    for i in range(10):
        if bin_dict[i]['count']:
            plt.text(rects[i].get_x(), rects[i].get_height(), bin_dict[i]['count'], va='bottom')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'ECE={ece:.4f}')
    plt.show()
    print('helpers/calibration_metrics.py')

if __name__ == '__main__':
    main()
