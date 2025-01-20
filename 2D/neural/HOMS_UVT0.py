# tuv0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import numpy as np
import os
import sys
import pandas as pd
import chaospy
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('N1.txt')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(torch.cuda.is_available())

class Block(nn.Module):

    def __init__(self, in_N, width, out_N):
        super(Block, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class drrnn(nn.Module):

    def __init__(self, in_N, m, depth, out_N):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()
        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        data_x, data_y, data_z = torch.chunk(x, 3, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * data_z * (1 - data_x) * (1 - data_y)


class drrnn_t(nn.Module):

    def __init__(self, in_N, m, depth, out_N):
        super(drrnn_t, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()
        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        data_x, data_y, data_z = torch.chunk(x, 3, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * data_z * (1 - data_x) * (1 - data_y) + 1

def get_interior_points_a(N,d):
    """
    N is the number of point, d is the dimension of point
    """
    return torch.rand(N,d)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def sort_xr(xr):
    column_index = 2
    sorted_values, indices = torch.sort(xr[:, column_index])
    sorted_matrix_by_column = xr[indices, :]
    return sorted_matrix_by_column

def fun(X):

    # u = t = torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt)
    # v = - torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt)

    c1111 = 16905.0078
    c1122 = 3711.5884
    c1212 = 6143.9546
    a11 = 57.6503
    b11 = 27.7521
    pc = 37.3613

    xrx, xry, xrt = torch.chunk(X, 3, 1)

    t_x = torch.pi * torch.cos(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt) * 1
    t_y = torch.pi * torch.sin(np.pi * xrx) * torch.cos(np.pi * xry) * torch.sin(torch.pi / 2 * xrt) * 1
    t_t = (torch.pi / 2) * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.cos(torch.pi / 2 * xrt) * 1
    t_xx = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt) * 1
    t_yy = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt) * 1

    u_xx = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt)
    u_xy = torch.pi ** 2 * torch.cos(np.pi * xrx) * torch.cos(np.pi * xry) * torch.sin(torch.pi / 2 * xrt)
    u_yy = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt)

    v_xx = torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt)
    v_xy = - torch.pi ** 2 * torch.cos(np.pi * xrx) * torch.cos(np.pi * xry) * torch.sin(torch.pi / 2 * xrt)
    v_yy = torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(torch.pi / 2 * xrt)

    y1 = - c1111 * u_xx - c1122 * v_xy - c1212 * u_yy - c1212 * v_xy + b11 * t_x
    y2 = - c1212 * u_xy - c1212 * v_xx - c1122 * u_xy - c1111 * v_yy + b11 * t_y
    h = pc * t_t - a11 * (t_xx + t_yy)

    return y1, y2, h

def quasirandom(n_samples, sampler):

    distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    if sampler == "halton":
        samples = distribution.sample(n_samples, rule="halton").T
        samples = torch.tensor(samples, dtype=torch.float)

    elif sampler == "hammersley":
        samples = distribution.sample(n_samples, rule="hammersley").T
        samples = torch.tensor(samples, dtype=torch.float)

    elif sampler == "additive_recursion":
        samples = distribution.sample(n_samples, rule="additive_recursion").T
        samples = torch.tensor(samples, dtype=torch.float)

    elif sampler == "korobov":
        samples = distribution.sample(n_samples, rule="korobov").T
        samples = torch.tensor(samples, dtype=torch.float)

    elif sampler == "sobol":
        samples = distribution.sample(n_samples, rule="sobol").T
        samples = torch.tensor(samples, dtype=torch.float)

    elif sampler == "rand":
        samples = torch.rand(n_samples, 2)

    elif sampler == "grid":
        N = int(n_samples ** (1 / 2))
        x1 = torch.linspace(0, 1, N)
        x2 = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x1, x2)
        samples = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)

    return samples


def main():

    c1111 = 16905.0078
    c1122 = 3711.5884
    c1212 = 6143.9546
    a11 = 57.6503
    b11 = 27.7521
    pc = 37.3613

    f1 = 10000
    f2 = 10000
    hh = 1000
    epochs = 30000
    lamb = 1
    epsilon = 2.8329e-10

    in_N = 3
    m = 20
    depth = 3
    out_N = 1

    path = r'E:\pycharm\project6\2d\uv\\'

    model_u = drrnn(in_N, m, depth, out_N).to(device)
    model_u.apply(weights_init)
    model_v = drrnn(in_N, m, depth, out_N).to(device)
    model_v.apply(weights_init)
    model_t = drrnn_t(in_N, m, depth, out_N).to(device)
    model_t.apply(weights_init)
    optimizer_u = optim.Adam(model_u.parameters(), lr=1e-2)
    scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, 500, gamma=0.9)
    optimizer_v = optim.Adam(model_v.parameters(), lr=1e-2)
    scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, 500, gamma=0.9)
    optimizer_t = optim.Adam(model_t.parameters(), lr=1e-2)
    scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_t, 500, gamma=0.9)
    print(model_u)

    # generate the data set
    xr = quasirandom(20000, "sobol")
    xr = sort_xr(xr)
    xr = xr.to(device)
    xr.requires_grad_()

    best_loss, best_epoch = torch.inf, 0
    stat = time.time()
    for epoch in range(epochs+1):

        xr_x, xr_y, xr_t = torch.chunk(xr, 3, 1)
        U = model_u(torch.cat((xr_x, xr_y, xr_t), dim=1))
        V = model_v(torch.cat((xr_x, xr_y, xr_t), dim=1))
        T = model_t(torch.cat((xr_x, xr_y, xr_t), dim=1))

        t_x = autograd.grad(outputs=T, inputs=xr_x,
                              grad_outputs=torch.ones_like(T),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        t_y = autograd.grad(outputs=T, inputs=xr_y,
                              grad_outputs=torch.ones_like(T),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        t_t = autograd.grad(outputs=T, inputs=xr_t,
                              grad_outputs=torch.ones_like(T),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        t_xx = autograd.grad(outputs=t_x, inputs=xr_x,
                              grad_outputs=torch.ones_like(t_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        t_yy = autograd.grad(outputs=t_y, inputs=xr_y,
                              grad_outputs=torch.ones_like(t_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        u_x = autograd.grad(outputs=U, inputs=xr_x,
                              grad_outputs=torch.ones_like(U),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_y = autograd.grad(outputs=U, inputs=xr_y,
                              grad_outputs=torch.ones_like(U),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_xx = autograd.grad(outputs=u_x, inputs=xr_x,
                              grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_yy = autograd.grad(outputs=u_y, inputs=xr_y,
                              grad_outputs=torch.ones_like(u_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_xy = autograd.grad(outputs=u_x, inputs=xr_y,
                              grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        v_x = autograd.grad(outputs=V, inputs=xr_x,
                              grad_outputs=torch.ones_like(V),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_y = autograd.grad(outputs=V, inputs=xr_y,
                              grad_outputs=torch.ones_like(V),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_xx = autograd.grad(outputs=v_x, inputs=xr_x,
                              grad_outputs=torch.ones_like(v_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_yy = autograd.grad(outputs=v_y, inputs=xr_y,
                              grad_outputs=torch.ones_like(v_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_xy = autograd.grad(outputs=v_x, inputs=xr_y,
                              grad_outputs=torch.ones_like(v_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        loss_r1 = (- (c1111 * u_xx + c1122 * v_xy + c1212 * u_yy + c1212 * v_xy) + b11 * t_x - f1) ** 2
        loss_r2 = (- (c1212 * u_xy + c1212 * v_xx + c1122 * u_xy + c1111 * v_yy) + b11 * t_y - f2) ** 2
        loss_r3 = (pc * t_t - a11 * (t_xx + t_yy) - hh) ** 2
        loss_r = loss_r1 + loss_r2 + loss_r3
        # cumulative_sum = torch.exp(- epsilon * torch.cumsum(loss_r, dim=0))
        # loss = torch.mean(loss_r * cumulative_sum)
        loss = torch.mean(loss_r)

        optimizer_u.zero_grad()
        optimizer_v.zero_grad()
        optimizer_t.zero_grad()
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        optimizer_u.step()
        scheduler_u.step()
        optimizer_v.step()
        scheduler_v.step()
        optimizer_t.step()
        scheduler_t.step()

        if epoch % 100 == 0:

            print('epoch:', epoch, 'loss:', loss.item(),'lr', optimizer_u.state_dict()['param_groups'][0]['lr'])

        if epoch > int(4 * epochs / 5):
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                torch.save(model_u.state_dict(), 'model_u0.mdl')
                torch.save(model_v.state_dict(), 'model_v0.mdl')
                torch.save(model_t.state_dict(), 'model_t0.mdl')

        if epoch % 500 == 0:

            x1 = torch.linspace(0, 1, 200)
            x2 = torch.linspace(0, 1, 200)
            X, Y = torch.meshgrid(x1, x2)
            Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None], Y.T.flatten()[:, None] * 0 + 0.05), dim=1)
            Z = Z.to(device)
            pre_u = model_u(Z)
            pre_v = model_v(Z)
            pre_t = model_t(Z)
            pred = pre_u.cpu().detach().numpy()
            pred_u2 = pred.reshape(200, 200)
            pred = pre_v.cpu().detach().numpy()
            pred_v2 = pred.reshape(200, 200)
            pred = pre_t.cpu().detach().numpy()
            pred_t2 = pred.reshape(200, 200)

            Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None], Y.T.flatten()[:, None] * 0 + 0.1), dim=1)
            Z = Z.to(device)
            pre_u = model_u(Z)
            pre_v = model_v(Z)
            pre_t = model_t(Z)
            pred = pre_u.cpu().detach().numpy()
            pred_u5 = pred.reshape(200, 200)
            pred = pre_v.cpu().detach().numpy()
            pred_v5 = pred.reshape(200, 200)
            pred = pre_t.cpu().detach().numpy()
            pred_t5 = pred.reshape(200, 200)

            plt.figure(figsize=(14, 8))
            plt.subplot(2, 3, 1)
            h = plt.imshow(pred_u2, interpolation='nearest', cmap='rainbow',
                           extent=[0, 1, 0, 1],
                           origin='lower', aspect='auto')
            plt.colorbar(h)
            plt.title("predict")

            plt.subplot(2, 3, 2)
            h = plt.imshow(pred_v2, interpolation='nearest', cmap='rainbow',
                           extent=[0, 1, 0, 1],
                           origin='lower', aspect='auto')
            plt.colorbar(h)
            plt.title("exact")

            plt.subplot(2, 3, 3)
            h = plt.imshow(pred_t2, interpolation='nearest', cmap='rainbow',
                           extent=[0, 1, 0, 1],
                           origin='lower', aspect='auto')
            plt.colorbar(h)
            plt.title("error")

            plt.subplot(2, 3, 4)
            h = plt.imshow(pred_u5, interpolation='nearest', cmap='rainbow',
                           extent=[0, 1, 0, 1],
                           origin='lower', aspect='auto')
            plt.colorbar(h)
            plt.title("predict")

            plt.subplot(2, 3, 5)
            h = plt.imshow(pred_v5, interpolation='nearest', cmap='rainbow',
                           extent=[0, 1, 0, 1],
                           origin='lower', aspect='auto')
            plt.colorbar(h)
            plt.title("exact")

            plt.subplot(2, 3, 6)
            h = plt.imshow(pred_t5, interpolation='nearest', cmap='rainbow',
                           extent=[0, 1, 0, 1],
                           origin='lower', aspect='auto')
            plt.colorbar(h)
            plt.title("error")

            plt.title(f"Epoch : {epoch}")
            plt.savefig(rf'{path}epoch_{epoch}.png')
            plt.clf()

    print("Running time: ", time.time()-stat)
    print('best epoch:', best_epoch, 'best loss:', best_loss)


if __name__ == '__main__':
    main()
