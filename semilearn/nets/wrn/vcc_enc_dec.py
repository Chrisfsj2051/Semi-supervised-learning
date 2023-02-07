import torch.nn as nn
import torch


class VCCBaseEncoderDecoder(nn.Module):
    def __init__(self, args, base_net):
        super(VCCBaseEncoderDecoder, self).__init__()
        self.args = args
        self.sampling_times = args.vcc_mcdropout_sampling_times
        self.num_classes = args.num_classes
        self.z_dim = args.vcc_z_dim
        self.base_net_channels = base_net.channels
        self.enc_with_bn = args.vcc_enc_norm in ['bn', 'bn+ln']
        self.enc_with_ln = args.vcc_enc_norm in ['ln', 'bn+ln']
        self.dec_with_bn = args.vcc_dec_norm in ['bn', 'bn+ln']
        self.dec_with_ln = args.vcc_dec_norm in ['ln', 'bn+ln']
        self.model = self.build_model()

    def build_simple_mlp(self, dims, with_bn):
        results = []
        for i in range(len(dims) - 1):
            results.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                if with_bn == 'bn':
                    results.append(nn.BatchNorm2d(dims[i + 1]))
                results.append(nn.ReLU(inplace=True))
        return results

class VCCEarlyFusionEncoder(VCCBaseEncoderDecoder):
    def build_model(self):
        if self.enc_with_ln:
            self.logits_ln = nn.LayerNorm(self.num_classes)
            self.feats_ln = nn.LayerNorm(self.base_net_channels)
        encoder_dims = ([self.num_classes + self.base_net_channels] +
                        self.args.vcc_encoder_dims + [2 * self.z_dim])
        return nn.Sequential(*self.build_simple_mlp(encoder_dims, self.enc_with_bn))

    def forward(self, logits, feats):
        if self.enc_with_ln:
            logits = self.logits_ln(logits)
            feats = self.feats_ln(feats)
        encoder_x = torch.cat([logits, feats], 1)
        if self.args.vcc_detach_input:
            encoder_x = encoder_x.detach()
        return self.model(encoder_x)

class VCCEarlyFusionDecoder(VCCBaseEncoderDecoder):
    def build_model(self):
        if self.dec_with_ln:
            self.logits_ln = nn.LayerNorm(self.num_classes)
            self.feats_ln = nn.LayerNorm(self.base_net_channels)
        decoder_dims = ([self.num_classes + self.base_net_channels + self.z_dim] +
                        self.args.vcc_decoder_dims + [self.num_classes])
        return nn.Sequential(*self.build_simple_mlp(decoder_dims, self.dec_with_bn))

    def forward(self, logits, feats, z):
        if self.dec_with_ln:
            logits = self.logits_ln(logits)
            feats = self.feats_ln(feats)
        decoder_x = torch.cat([logits, feats, z], 1)
        if self.args.vcc_detach_input:
            decoder_x = decoder_x.detach()
        return self.model(decoder_x)


class VCCLateFusionDecoder(VCCBaseEncoderDecoder):
    def build_model(self):
        def build_single_branch(dims, with_bn, with_ln):
            results = self.build_simple_mlp(dims + [1], with_bn)[:-1]
            results = ([nn.LayerNorm(dims[0])] if with_ln else []) + results
            return results

        dims = self.args.vcc_decoder_dims
        logits_branch = build_single_branch(
            [self.num_classes] + dims, self.dec_with_bn, self.dec_with_ln)
        feats_branch = build_single_branch(
            [self.base_net_channels] + dims, self.dec_with_bn, self.dec_with_ln)
        merged_branch = build_single_branch(
            [dims[-1] * 2 + self.z_dim, self.num_classes], False, False)

        return nn.ModuleDict({
            'logits': nn.Sequential(*logits_branch),
            'feats': nn.Sequential(*feats_branch),
            'merged': nn.Sequential(*merged_branch)
        })

    def forward(self, logits, feats, z):
        if self.args.vcc_detach_input:
            # TODO: detach z?
            logits = logits.detach()
            feats = feats.detach()

        logits = self.model['logits'](logits)
        feats = self.model['feats'](feats)
        merged = torch.cat([logits, feats, z], 1)
        return self.model['merged'](merged)
