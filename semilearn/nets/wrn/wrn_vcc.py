from semilearn.nets.wrn.wrn import wrn_28_2, wrn_28_8
import torch.nn as nn
import torch
import torch.nn.functional as F


class VariationalConfidenceCalibration(nn.Module):

    def __init__(self, base_net, args, num_classes):
        super(VariationalConfidenceCalibration, self).__init__()
        self.sampling_times = 20
        self.num_classes = num_classes
        self.base_net = base_net
        self.z_dim = args.vcc_z_dim
        self.encoder = nn.Sequential(
            nn.Linear(num_classes + base_net.channels, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_classes + base_net.channels + self.z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def calc_uncertainty(self, img, feats):
        batch_size = feats.shape[0]
        num_classes = self.num_classes
        feats = torch.cat([feats for _ in range(self.sampling_times)], 0)
        with torch.no_grad():
            feats = torch.dropout(feats, p=0.5, train=True)
            pred = self.base_net.fc(feats).argmax(1)

        pred_onehot = F.one_hot(pred, num_classes)
        pred_onehot = pred_onehot.reshape(self.sampling_times, batch_size, num_classes)
        pred_onehot = pred_onehot.permute(1, 0, 2)
        pred_onehot = pred_onehot.sum(1).float() / self.sampling_times
        return pred_onehot

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        assert not only_fc
        assert not only_feat
        backbone_output = self.base_net(x, only_fc, only_feat, **kwargs)
        logits, feats = backbone_output['logits'], backbone_output['feat']

        cali_gt_label = self.calc_uncertainty(x, feats)
        encoder_x = torch.cat([logits, feats], 1)
        h = self.encoder(encoder_x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterise(mu, logvar)
        recon_r = self.decoder(torch.cat([logits, feats, z], 1))

        with torch.no_grad():
            h = torch.randn(x.shape[0], self.z_dim * 2).to(x.device)
            sample_mu, sample_logvar = h.chunk(2, dim=1)
            z = self.reparameterise(sample_mu, sample_logvar)
            decode_input = torch.cat([logits, feats, z], 1)
            cali_output = self.decoder(decode_input)

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
