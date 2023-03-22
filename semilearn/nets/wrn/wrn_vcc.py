import numpy as np

from semilearn.nets.wrn.wrn import wrn_28_2, wrn_28_8
from semilearn.nets.wrn.wrn_var import wrn_var_37_2
from semilearn.nets.wrn.vcc_enc_dec import VCCEarlyFusionEncoder, VCCEarlyFusionDecoder
from semilearn.nets.plugins.dropblock import DropBlock2D
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist
import joblib


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
        self.decoder = VCCEarlyFusionDecoder(args, base_net)

        # placeholder
        self.history_preds = None
        self.datapoint_bank = None

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def update_save_dict(self, save_dict):
        if self.history_preds is not None:
            save_dict['vcc_history_preds'] = self.history_preds
        if self.datapoint_bank is not None:
            save_dict['vcc_datapoint_bank'] = self.datapoint_bank
        return save_dict

    def load_model(self, checkpoint):
        if 'vcc_history_preds' in checkpoint.keys():
            self.history_preds = checkpoint['vcc_history_preds']
        if 'vcc_datapoint_bank' in checkpoint.keys():
            self.datapoint_bank = checkpoint['vcc_datapoint_bank']

    def calc_uncertainty_consistency(self, algorithm, x, ulb_x_idx, feats, logits, batch_size, num_classes):
        assert self.args.vcc_lab_loss_weight == 0
        ulb_num, lb_num = ulb_x_idx.shape[0], batch_size - 2 * ulb_x_idx.shape[0]
        lb_x, (ulb_x_w, ulb_x_s) = x[:lb_num], x[lb_num:].chunk(2)
        total_ulb_num = len(algorithm.dataset_dict['train_ulb'])
        if self.datapoint_bank is None:
            self.datapoint_bank = [[] for _ in range(num_classes)]

        all_confidence = logits.softmax(1).detach()
        preds = all_confidence[lb_num:lb_num + ulb_num]

        # Temporal Consistency
        if self.history_preds is None:
            self.history_preds = ulb_x_s.new_ones((total_ulb_num, num_classes)) / num_classes
        self.history_preds = self.history_preds.to(ulb_x_s.device)

        prev_preds = self.history_preds[ulb_x_idx]
        # if (abs(self.history_preds.sum(1) - 1.0) > 1e-5).any():
        #     joblib.dump(self.history_preds.cpu().detach().numpy(), 'debug.pth')
        #     print('Save debug')
        # assert abs(prev_preds[0].sum() - 1.0) < 1e-5
        temporal_kl_div = torch.kl_div((preds + 1e-7).log(), prev_preds).sum(1)
        upd_preds = ulb_x_s.new_zeros((total_ulb_num, num_classes))
        upd_cnt = ulb_x_s.new_zeros((total_ulb_num,))
        upd_preds[ulb_x_idx], upd_cnt[ulb_x_idx] = preds, 1
        if algorithm.args.distributed:
            dist.all_reduce(upd_preds, op=dist.ReduceOp.SUM)
            dist.all_reduce(upd_cnt, op=dist.ReduceOp.SUM)
            dist.barrier()
        upd_mask = (upd_cnt != 0)
        upd_preds[upd_mask] /= upd_cnt[upd_mask][:, None]
        self.history_preds[upd_mask] = upd_preds[upd_mask]

        # Instance Consistency
        self.dropout_keep_p, self.sampling_times = 0.7, 8
        _, ic_logits = self.calc_uncertainty_mcdropout(feats[lb_num:lb_num + ulb_num], ulb_num, num_classes)
        ic_pred = ic_logits.softmax(1).reshape(self.sampling_times, ulb_num, num_classes)
        ic_pred = ic_pred.mean(0)
        entropy = -(ic_pred * (ic_pred + 1e-7).log()).sum(1)

        # View Consistency: EMA v.s. ori
        algorithm.ema.apply_shadow()
        ema_feats = self.base_net(ulb_x_w, only_feat=True)
        ema_logits = self.base_net(feats[lb_num:lb_num + ulb_num], only_fc=True)
        ema_preds = ema_logits.softmax(1)
        algorithm.ema.restore()
        ori_logits = self.base_net(ema_feats, only_fc=True)
        ori_preds = ori_logits.softmax(1)
        view_kl_div = torch.kl_div((ori_preds + 1e-7).log(), ema_preds).sum(1)

        # Ori confidence
        confidence = all_confidence[lb_num:lb_num + ulb_num]

        if algorithm.args.distributed:
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1
        gmm_feats = torch.cat(
            [confidence.max(1)[0][None], temporal_kl_div[None], entropy[None], view_kl_div[None]], 0
        ).transpose(0, 1)
        pseudo_labels = logits[lb_num:lb_num + ulb_num].argmax(1)
        dist_gmm_feats = gmm_feats.new_zeros(ulb_num * world_size, gmm_feats.shape[1])
        dist_pseudo_labels = pseudo_labels.new_zeros(ulb_num * world_size)
        dist_gmm_feats[ulb_num * rank: ulb_num * (rank + 1)] = gmm_feats
        dist_pseudo_labels[ulb_num * rank: ulb_num * (rank + 1)] = pseudo_labels
        if algorithm.args.distributed:
            dist.all_reduce(dist_gmm_feats, op=dist.ReduceOp.SUM)
            dist.all_reduce(dist_pseudo_labels, op=dist.ReduceOp.SUM)
            dist.barrier()
        for i, label in enumerate(dist_pseudo_labels):
            self.datapoint_bank[label].append(dist_gmm_feats[i].cpu().tolist())
            self.datapoint_bank[label] = self.datapoint_bank[label][-100:]

        cali_conf = all_confidence

        def compute_score(data, max_norm, min_norm):
            max_norm, min_norm = max_norm[None], min_norm[None]
            data = (data - min_norm) / (max_norm - min_norm + 1e-5)
            data[:, 0] = 1 - data[:, 0]
            return (data ** 2).sum(1)

        for label in set(pseudo_labels.tolist()):
            if len(self.datapoint_bank[label]) < 50:
                continue
            mask = (pseudo_labels == label)
            cls_data = np.array(self.datapoint_bank[label])
            max_norm, min_norm = cls_data.max(0), cls_data.min(0)
            max_conf, min_conf = max_norm[0], min_norm[0]
            cls_score = compute_score(cls_data, max_norm, min_norm)
            max_score, min_score = cls_score.max(), cls_score.min()
            batch_score = compute_score(gmm_feats[mask].cpu().detach().numpy(), max_norm, min_norm)
            batch_cali_conf = ((max_score - batch_score) / (max_score - min_score + 1e-7)
                               * (max_conf - min_conf) + min_conf)
            batch_cali_conf = cali_conf.new_tensor(batch_cali_conf)
            ori_confidence = confidence[mask]
            ori_others_conf = 1 - ori_confidence[:, label]
            cur_others_conf = 1 - batch_cali_conf
            cali_conf[lb_num:ulb_num+lb_num][mask] *= (
                    cur_others_conf / (ori_others_conf + 1e-7))[..., None]
            cali_conf[lb_num:ulb_num+lb_num][mask, label] = batch_cali_conf

        return cali_conf, None

    def calc_uncertainty_mcdropout(self, feats, batch_size, num_classes, **kwargs):
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

    def calc_uncertainty_mcdropout_mean(self, feats, batch_size, num_classes, **kwargs):
        _, pred = self.calc_uncertainty_mcdropout(feats, batch_size, num_classes)
        result = pred.reshape(self.sampling_times, batch_size, num_classes)
        result = result.softmax(2)
        result = result.mean(0)
        return result, pred

    def calc_uncertainty_mccutout(self, x, batch_size, num_classes, **kwargs):
        dropblock = DropBlock2D(1 - self.args.vcc_mc_keep_p, self.args.vcc_mc_dropsize)
        img = dropblock(torch.cat([x for _ in range(self.sampling_times)], 0))
        with torch.no_grad():
            pred = self.base_net(img)['logits']
        result = pred.argmax(1)
        result = F.one_hot(result, num_classes)
        result = result.reshape(self.sampling_times, batch_size, num_classes)
        result = result.permute(1, 0, 2)
        result = result.sum(1).float() / self.sampling_times
        return result, pred

    def eval(self):
        super(VariationalConfidenceCalibration, self).eval()
        self.train(False)

    def train(self, mode):
        super(VariationalConfidenceCalibration, self).train(mode)
        self.base_net.train(mode)
        self.decoder.train(mode)
        self.encoder.train(mode)

    def calc_uncertainty(self, **kwargs):
        kwargs['batch_size'] = kwargs['feats'].shape[0]
        kwargs['num_classes'] = self.num_classes
        uncertainty_method = self.args.vcc_uncertainty_method
        uncertainty_method = getattr(self, f'calc_uncertainty_{uncertainty_method}')
        return uncertainty_method(**kwargs)[0]

    def forward(self, algorithm, x, only_fc=False, only_feat=False, ulb_x_idx=None, **kwargs):
        assert not only_fc
        assert not only_feat
        backbone_output = self.base_net(x, only_fc, only_feat, **kwargs)
        logits, feats = backbone_output['logits'], backbone_output['feat']
        if ulb_x_idx is not None:
            cali_gt_label = self.calc_uncertainty(
                algorithm=algorithm, x=x, ulb_x_idx=ulb_x_idx, feats=feats, logits=logits)
        else:
            cali_gt_label = None
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
            #     print('conf:', logits.softmax(1)[idx].topk(1), '\n')
            #     print('cali_gt:', cali_gt_label[idx].topk(1), '\n')
            #     print('cali_recon:', recon_r.softmax(1)[idx].topk(1), '\n')
            #     print('cali_repa_recon:')
            #     for _ in range(3):
            #         z = self.reparameterise(sample_mu, sample_logvar)
            #         cali_output = self.decoder(x, logits, feats, z)
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
