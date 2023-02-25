from semilearn.nets.wrn.wrn import wrn_28_2, wrn_28_8
from semilearn.nets.wrn.wrn_var import wrn_var_37_2
from semilearn.nets.wrn.vcc_enc_dec import VCCEarlyFusionEncoder, VCCEarlyFusionDecoder
from semilearn.nets.plugins.dropblock import DropBlock2D
import torch.nn as nn
import torch
import torch.nn.functional as F


class VariationalConfidenceCalibration(nn.Module):

    def __init__(self, base_net, args, num_classes):
        super(VariationalConfidenceCalibration, self).__init__()
        self.args = args
        self.sampling_times = args.vcc_mc_sampling_times
        self.num_classes = num_classes
        self.base_net = base_net
        self.z_dim = args.vcc_z_dim
        self.dropout_keep_p = args.vcc_mc_keep_p
        self.detach_input = args.vcc_detach_input
        self.encoder = VCCEarlyFusionEncoder(args, base_net)
        decoder_type = VCCEarlyFusionDecoder
        # if args.vcc_dec_model == 'late_fusion':
        #     decoder_type = VCCLateFusionDecoder
        self.decoder = decoder_type(args, base_net)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def calc_uncertainty_mcdropout(self, img, feats, batch_size, num_classes):
        feats = torch.cat([feats for _ in range(self.sampling_times)], 0)
        with torch.no_grad():
            feats = torch.dropout(feats, p=1 - self.dropout_keep_p, train=True)
            pred = self.base_net.fc(feats)
        result = pred.argmax(1)
        result = F.one_hot(result, num_classes)
        result = result.reshape(self.sampling_times, batch_size, num_classes)
        result = result.permute(1, 0, 2)
        result = result.sum(1).float() / self.sampling_times
        return result, pred

    def calc_uncertainty_mcdropout_mean(self, img, feats, batch_size, num_classes):
        _, pred = self.calc_uncertainty_mcdropout(img, feats, batch_size, num_classes)
        result = pred.reshape(self.sampling_times, batch_size, num_classes)
        result = result.softmax(2)
        result = result.mean(0)
        return result, pred

    def calc_uncertainty_mccutout(self, img, feats, batch_size, num_classes):
        dropblock = DropBlock2D(1 - self.args.vcc_mc_keep_p, self.args.vcc_mc_dropsize)
        img = dropblock(torch.cat([img for _ in range(self.sampling_times)], 0))
        with torch.no_grad():
            pred = self.base_net(img)['logits']
        result = pred.argmax(1)
        result = F.one_hot(result, num_classes)
        result = result.reshape(self.sampling_times, batch_size, num_classes)
        result = result.permute(1, 0, 2)
        result = result.sum(1).float() / self.sampling_times
        return result, pred

    def calc_uncertainty(self, img, feats):
        batch_size = feats.shape[0]
        num_classes = self.num_classes
        uncertainty_method = self.args.vcc_uncertainty_method
        uncertainty_method = getattr(self, f'calc_uncertainty_{uncertainty_method}')
        return uncertainty_method(img, feats, batch_size, num_classes)[0]

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        assert not only_fc
        assert not only_feat
        backbone_output = self.base_net(x, only_fc, only_feat, **kwargs)
        logits, feats = backbone_output['logits'], backbone_output['feat']
        cali_gt_label = self.calc_uncertainty(x, feats)
        h = self.encoder(x, logits, feats)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterise(mu, logvar)
        recon_r = self.decoder(x, logits, feats, z)  # train vcc

        with torch.no_grad():
            h = torch.randn(x.shape[0], self.z_dim * 2).to(x.device)
            sample_mu, sample_logvar = h.chunk(2, dim=1)
            z = self.reparameterise(sample_mu, sample_logvar)
            cali_output = self.decoder(x, logits, feats, z)
            # for idx in range(self.args.batch_size, self.args.batch_size * 8):
            #     import math
            #     if math.fabs(recon_r.softmax(1)[idx].max() - cali_gt_label[idx].max()) < 0.05:
            #         continue
            #     print(cali_gt_label[idx].topk(1), '\n')
            #     for _ in range(3):
            #         z = self.reparameterise(sample_mu, sample_logvar)
            #         cali_output = self.decoder(logits, feats, z)
            #         print(cali_output.softmax(1)[idx].topk(1))
            #     print('==============')

        return {
            'logits': logits,
            'recon_pred': recon_r,
            'recon_gt': cali_gt_label,
            'mu': mu,
            'logvar': logvar,
            'calibrated_logits': cali_output
        }


def vcc_wrn_28_2(pretrained=False, pretrained_path=None, args=None, num_classes=0, **kwargs):
    base_model = wrn_28_2(pretrained, pretrained_path, args=args, num_classes=num_classes, **kwargs)
    model = VariationalConfidenceCalibration(base_model, args, num_classes)
    assert not pretrained_path
    return model


def vcc_wrn_28_8(pretrained=False, pretrained_path=None, args=None, num_classes=0, **kwargs):
    base_model = wrn_28_8(pretrained, pretrained_path, args=args, num_classes=num_classes, **kwargs)
    model = VariationalConfidenceCalibration(base_model, args, num_classes)
    assert not pretrained_path
    return model


def vcc_wrn_var_37_2(pretrained=False, pretrained_path=None, args=None, num_classes=0, **kwargs):
    base_model = wrn_var_37_2(pretrained, pretrained_path, args=args, num_classes=num_classes, **kwargs)
    model = VariationalConfidenceCalibration(base_model, args, num_classes)
    assert not pretrained_path
    return model
