import torch
import matplotlib.pyplot as plt
import joblib
from semilearn.core.utils.calibration_metrics import _populate_bins, BIN_ACC, expected_calibration_error
from tools.visualize_confidence import init_algorithm
from tools.helpers import ReliabilityDiagram, ConfidenceHistogram


@torch.no_grad()
def main():
    algorithm = init_algorithm()
    if algorithm.args.visualize_eval_results_path:
        result = joblib.load(algorithm.args.visualize_eval_results_path)
        y_true = result['y_true']
        y_pred = result['y_pred']
        y_probs = result['y_probs']
        y_logits = result['y_logits']
    else:
        (y_true, y_pred, y_logits, y_probs, total_loss, total_num
         ) = algorithm.predict('eval')
    calibration_diag = ConfidenceHistogram()
    calibration_hist = ReliabilityDiagram()
    plt = calibration_hist.plot(y_logits, y_true)
    plt.savefig('exchange/fully_cali_hist.png')
    # plt.show()
    plt = calibration_diag.plot(y_logits, y_true)
    plt.savefig('exchange/fully_cali_diag.png')
    # plt.show()

if __name__ == '__main__':
    main()
