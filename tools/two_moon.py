from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter1d

def draw_two_moon(X_0_lb, X_0_ulb, X_1_lb, X_1_ulb, model):
    plt.scatter(X_0_ulb[:, 0], X_0_ulb[:, 1], c='blue', marker='.')
    plt.scatter(X_0_lb[:, 0], X_0_lb[:, 1], c='blue', marker='+', s=150.0)
    plt.scatter(X_1_ulb[:, 0], X_1_ulb[:, 1], c='red', marker='.')
    plt.scatter(X_1_lb[:, 0], X_1_lb[:, 1], c='red', marker='^', s=150.0)
    plt.show()
    return

    all_X = torch.Tensor(np.concatenate([X_0_lb, X_0_ulb, X_1_lb, X_1_ulb], 0))
    # all_out = model(all_X)
    # errors = torch.abs(all_out - 0.5)
    # all_X = torch.cat([all_X, errors], 1)
    # _, indices = torch.topk(-all_X[:, 2], k=100)
    # p_x = all_X[indices, 0].tolist()
    # p_y = all_X[indices, 1].tolist()


    axis = torch.Tensor(list(np.arange(-2.0, 2.0, 0.005)))
    x, y = torch.meshgrid(axis, axis)
    grid = torch.cat([x[..., None], y[..., None]], -1).view(-1, 2)
    # dis = ((grid[:, None, :] - all_X[None]) ** 2).sum(2).min(1)[0]
    # errors = torch.abs(model(grid) - 0.5).squeeze()
    errors = (model(grid) - 0.5).squeeze()
    pseudo_label = errors > 0
    plt.scatter(grid[pseudo_label, 0], grid[pseudo_label, 1], c='blue')
    plt.scatter(grid[~pseudo_label, 0], grid[~pseudo_label, 1], c='red')

    errors = errors.reshape(axis.shape[0], axis.shape[0])
    optim_y = axis[errors.argmin(1)]
    p_x, p_y = axis.tolist(), optim_y.tolist()
    # for i in range(5, len(p_y)):
    #     tmp = 0
    #     for _ in range(5):
    #         tmp += p_y[i - _]
    #     p_y[i] = tmp / 5.0
    # plt.plot(p_x, p_y)
    plt.show()



seed = 1
total_samples = 300
lb_samples_per_cls = 20
epoch = 100
lb_batch_size = 4
ulb_batch_size = lb_batch_size * 4

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

X, Y = make_moons(n_samples=total_samples, noise=0.1)
X_0, X_1 = X[Y == 0], X[Y == 1]
X_0_lb, X_0_ulb = X_0[:lb_samples_per_cls], X_0[lb_samples_per_cls:]
X_1_lb, X_1_ulb = X_1[:lb_samples_per_cls], X_1[lb_samples_per_cls:]

model = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
    # nn.Linear(2, 1),
    nn.Sigmoid()
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for e in range(epoch):
    X_lb = [(x[0], x[1], 0) for x in X_0_lb] + [(x[0], x[1], 1) for x in X_1_lb]
    X_ulb = X_1_ulb + X_0_ulb
    # random.shuffle(X_ulb)
    # random.shuffle(X_lb)
    Y_lb_all = torch.Tensor(X_lb)[:, 2]
    X_lb_all = torch.Tensor(X_lb)[:, :2]
    X_ulb_all = torch.Tensor(X_ulb)
    for X_lb, Y_lb, X_ulb in zip(
        X_lb_all.chunk(X_lb_all.shape[0] // lb_batch_size),
        Y_lb_all.chunk(X_lb_all.shape[0] // lb_batch_size),
        X_ulb_all.chunk(X_ulb_all.shape[0] // ulb_batch_size)
    ):
        X_concat = torch.cat([X_lb, X_ulb], 0)
        num_lb = X_lb.shape[0]
        out_concat = model(X_concat)
        lb_loss = nn.BCELoss()(out_concat[:num_lb, 0], Y_lb)
        mask = (out_concat[num_lb:] > 0.95)
        pseudo_label = (out_concat[num_lb:] > 0.5).float()
        ulb_loss = (nn.BCELoss(reduction='none')(out_concat[num_lb:], pseudo_label) * mask).sum(1).mean(0)
        loss = lb_loss + ulb_loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

    if e == 0 or (e + 1) % 10 == 0:
        print(e, loss.item())
        draw_two_moon(X_0_lb, X_0_ulb, X_1_lb, X_1_ulb, model)
        break
