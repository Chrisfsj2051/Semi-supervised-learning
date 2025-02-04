# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py

import os
import joblib

from .hook import Hook

class CheckpointHook(Hook):
    def __init__(self):
        super().__init__()

    def after_train_step(self, algorithm):
        if ((not algorithm.distributed) or
                (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0)):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)

            if (self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm)):
                algorithm.save_model('latest_model.pth', save_path)
                eval_results_path = os.path.join(save_path, 'eval_results')
                if not os.path.exists(eval_results_path):
                    os.makedirs(eval_results_path)
                joblib.dump(algorithm.eval_results,
                            os.path.join(eval_results_path, f'eval_{algorithm.it + 1:010d}.pth'))

            if algorithm.it == algorithm.best_it:
                algorithm.save_model('model_best.pth', save_path)

            if 'vcc' in algorithm.algorithm and algorithm.it == algorithm.vcc_training_warmup:
                algorithm.save_model('model_vcc_training_init.pth', save_path)

            if (algorithm.it + 1) % algorithm.args.save_interval == 0:
                algorithm.save_model(f'model_{algorithm.it + 1:010d}.pth', save_path)
