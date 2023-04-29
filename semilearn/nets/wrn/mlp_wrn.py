import numpy as np

from semilearn.nets.utils import load_checkpoint
from semilearn.nets.wrn.wrn import wrn_28_2, wrn_28_8, WideResNet
from semilearn.nets.wrn.wrn_var import wrn_var_37_2
from semilearn.nets.wrn.vcc_enc_dec import VCCEarlyFusionEncoder, VCCEarlyFusionDecoder
from semilearn.nets.plugins.dropblock import DropBlock2D
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist
import joblib

class MLP_WideResNet(WideResNet):

    def __init__(self, *args, **kwargs):
        super(MLP_WideResNet, self).__init__(*args, **kwargs)
        self.select_model = nn.Sequential(
            nn.Linear(self.channels, 1)
        )

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """

        if only_fc:
            return self.fc(x) * self.temperature_scaling

        out = self.extract(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)

        if only_feat:
            return out

        output = self.fc(out) * self.temperature_scaling
        example_logits = self.select_model(out)

        result_dict = {'logits': output, 'feat': out, 'example_logits': example_logits}
        return result_dict



def mlp_wrn_28_2(pretrained=False, pretrained_path=None, args=None, **kwargs):
    model = MLP_WideResNet(first_stride=1, depth=28, widen_factor=2, args=args, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
