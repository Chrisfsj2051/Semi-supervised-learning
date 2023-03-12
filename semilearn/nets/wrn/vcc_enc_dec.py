import torch.nn as nn
import torch


class VCCBaseEncoderDecoder(nn.Module):
    def __init__(self, args, base_net):
        super(VCCBaseEncoderDecoder, self).__init__()
        self.args = args
        self.sampling_times = args.vcc_mc_sampling_times
        self.num_classes = args.num_classes
        self.z_dim = args.vcc_z_dim
        self.base_net_channels = base_net.channels
        self.enc_with_bn = args.vcc_enc_norm in ['bn', 'bn+ln']
        self.enc_with_ln = args.vcc_enc_norm in ['ln', 'bn+ln']
        self.dec_with_bn = args.vcc_dec_norm in ['bn', 'bn+ln']
        self.dec_with_ln = args.vcc_dec_norm in ['ln', 'bn+ln']

    def build_simple_mlp(self, dims, with_bn, activation):
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'elu':
            activation = nn.ELU
        results = []
        for i in range(len(dims) - 1):
            results.append(nn.Linear(dims[i], dims[i + 1]))
            if with_bn:
                results.append(nn.BatchNorm1d(dims[i + 1]))
            results.append(activation(inplace=True))
        return results


class VCCEarlyFusionEncoder(VCCBaseEncoderDecoder):

    def __init__(self, args, base_net):
        super(VCCEarlyFusionEncoder, self).__init__(args, base_net)
        if self.enc_with_ln:
            self.logits_ln = nn.LayerNorm(self.num_classes)
            self.feats_ln = nn.LayerNorm(self.base_net_channels)
        shared_dims = ([self.num_classes + self.base_net_channels] +
                       self.args.vcc_encoder_dims)
        self.shared_branch = nn.Sequential(
            *self.build_simple_mlp(shared_dims, self.enc_with_bn, activation='elu'))
        self.mu_branch = nn.Sequential(
            nn.Linear(shared_dims[-1], self.z_dim),
            nn.ELU(inplace=True)
        )
        self.logvar_branch = nn.Sequential(
            nn.Linear(shared_dims[-1], self.z_dim),
            nn.ELU(inplace=True)
        )

    def forward(self, img, logits, feats):
        if self.enc_with_ln:
            logits = self.logits_ln(logits)
            feats = self.feats_ln(feats)
        encoder_x = torch.cat([logits, feats], 1)
        if self.args.vcc_detach_input:
            encoder_x = encoder_x.detach()
        shared_embedding = self.shared_branch(encoder_x)
        mu = self.mu_branch(shared_embedding)
        logvar = self.logvar_branch(shared_embedding)
        return torch.cat([mu, logvar], 1)


class VCCEarlyFusionDecoder(VCCBaseEncoderDecoder):
    def __init__(self, args, base_net):
        super(VCCEarlyFusionDecoder, self).__init__(args, base_net)
        if self.dec_with_ln:
            self.logits_ln = nn.LayerNorm(self.num_classes)
            self.feats_ln = nn.LayerNorm(self.base_net_channels)
        decoder_dims = ([self.num_classes + self.base_net_channels + self.z_dim] +
                        self.args.vcc_decoder_dims)
        self.model = nn.Sequential(
            *self.build_simple_mlp(decoder_dims, self.dec_with_bn, activation='relu'))
        self.fc = nn.Linear(decoder_dims[-1], self.num_classes)

    def forward(self, img, logits, feats, z):
        if self.dec_with_ln:
            logits = self.logits_ln(logits)
            feats = self.feats_ln(feats)
        decoder_x = torch.cat([logits, feats, z], 1)
        if self.args.vcc_detach_input:
            decoder_x = decoder_x.detach()
        embedding = self.model(decoder_x)
        return self.fc(embedding)
