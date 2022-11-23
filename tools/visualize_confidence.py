from tools.utils import build_model
import numpy as np
import matplotlib.pyplot as plt

def main():
    algorithm = build_model()
    algorithm.dataset_dict['train_ulb'].alg = 'evaluation'
    model = algorithm.model
    for data in algorithm.dataset_dict['train_ulb']:
        x_ulb_w, x_ulb_s, y_ulb = data['x_ulb_w'], data['x_ulb_s'], data['y_ulb']
        model.eval()
        model.ema.apply_shadow()
        eval_loader = self.loader_dict[eval_dest]
        total_loss, total_num = 0.0, 0.0
        y_true, y_pred, y_probs, y_logits = [], [], [], []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)['logits']

                loss = F.cross_entropy(logits, y, reduction='mean')
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                total_loss += loss.item() * num_batch

        self.ema.restore()
        self.model.train()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        y_probs = np.concatenate(y_probs)
        print('in')

if __name__ == '__main__':
    main()
