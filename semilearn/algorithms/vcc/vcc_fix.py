from .vcc import VCC
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.core import AlgorithmBase


class VCCFixMatch(VCC):

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(FixedThresholdingHook(), "MaskingHook")

    def get_save_dict(self):
        save_dict = AlgorithmBase.get_save_dict(self)
        save_dict = self.update_save_dict(save_dict)
        return save_dict

    def load_model(self, load_path):
        checkpoint = AlgorithmBase.load_model(self, load_path)
        checkpoint = self.load_vcc_model(checkpoint)
        return checkpoint