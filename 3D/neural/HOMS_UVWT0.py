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
        data_x, data_y, data_z, data_t = torch.chunk(x, 4, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * data_z * data_t * (1 - data_x) * (1 - data_y) * (1 - data_z)

class drrnnt(nn.Module):

    def __init__(self, in_N, m, depth, out_N):
        super(drrnnt, self).__init__()
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
        data_x, data_y, data_z, data_t = torch.chunk(x, 4, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * data_z * data_t * (1 - data_x) * (1 - data_y) * (1 - data_z) + 1

def get_interior_points_a(N,d):
    """
    N is the number of point, d is the dimension of point
    """
    return torch.rand(N,d)

def get_interior_points(N, d):
    r = 0.3
    all = torch.rand(N,d)
    index1 = torch.rand(int(N / 8), 1) * 2 * torch.pi
    index2 = torch.rand(int(N / 8), 1)
    inner = torch.cat((torch.sin(index1), torch.cos(index1)), dim=1) * r * index2 + 0.5
    inner_boundary = torch.cat((torch.sin(index1), torch.cos(index1)), dim=1) * ((index2 * 0.1) + (r - 0.05)) + 0.5
    return torch.cat((all, inner, inner_boundary),dim=0)

def get_init_points(N):

    index1 = torch.rand(N, 1)
    xb = torch.cat((torch.rand(N, 1), torch.rand(N, 1), torch.full_like(index1, 0)), dim=1)
    return xb

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def fun(X):

    # u = w = t = torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    # v = - torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)

    c1111 = 29470.53
    c1122 = 9259.38
    c1212 = 9953.60
    a11 = 84.8864
    b11 = 41.3111
    pc = 45.2249

    xrx, xry, xrz, xrt = torch.chunk(X, 4, 1)

    t_x = torch.pi * torch.cos(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt) * 1
    t_y = torch.pi * torch.sin(np.pi * xrx) * torch.cos(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt) * 1
    t_z = torch.pi * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.cos(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt) * 1
    t_t = (torch.pi / 2) * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.cos(torch.pi / 2 * xrt) * 1
    t_xx = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt) * 1
    t_yy = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt) * 1
    t_zz = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt) * 1

    u_xx = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    u_yy = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    u_zz = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    u_xy = torch.pi ** 2 * torch.cos(np.pi * xrx) * torch.cos(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    u_xz = torch.pi ** 2 * torch.cos(np.pi * xrx) * torch.sin(np.pi * xry) * torch.cos(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)

    v_xx = torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    v_yy = torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    v_zz = torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    v_xy = - torch.pi ** 2 * torch.cos(np.pi * xrx) * torch.cos(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    v_yz = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.cos(np.pi * xry) * torch.cos(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)

    w_xx = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    w_yy = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    w_zz = - torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.sin(np.pi * xry) * torch.sin(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    w_xz = torch.pi ** 2 * torch.cos(np.pi * xrx) * torch.sin(np.pi * xry) * torch.cos(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)
    w_yz = torch.pi ** 2 * torch.sin(np.pi * xrx) * torch.cos(np.pi * xry) * torch.cos(np.pi * xrz) * torch.sin(torch.pi / 2 * xrt)

    y1 = - c1111 * u_xx - c1122 * v_xy - c1122 * w_xz - c1212 * u_yy - c1212 * v_xy - c1212 * u_zz - c1212 * w_xz + b11 * t_x
    y2 = - c1212 * u_xy - c1212 * v_xx - c1122 * u_xy - c1111 * v_yy - c1122 * w_yz - c1212 * v_zz - c1212 * w_yz + b11 * t_y
    y3 = - c1212 * u_xz - c1212 * w_xx - c1212 * v_yz - c1212 * w_yy - c1122 * u_xz - c1122 * v_yz - c1111 * w_zz + b11 * t_z
    h = pc * t_t - a11 * (t_xx + t_yy + t_zz)

    return y1, y2, y3, h

def main():

    c1111 = 29470.53
    c1122 = 9259.38
    c1212 = 9953.60
    a11 = 84.8864
    b11 = 41.3111
    pc = 45.2249

    f1 = 10000
    f2 = 10000
    f3 = 10000
    hh = 1000

    epochs = 20000

    in_N = 4
    m = 20
    depth = 3
    out_N = 1

    path = r'E:\PyCharm\project6\thermo-mechanical\3D\test\uv\\'

    model_u = drrnn(in_N, m, depth, out_N).to(device)
    model_u.apply(weights_init)
    model_v = drrnn(in_N, m, depth, out_N).to(device)
    model_v.apply(weights_init)
    model_w = drrnn(in_N, m, depth, out_N).to(device)
    model_w.apply(weights_init)
    model_t = drrnnt(in_N, m, depth, out_N).to(device)
    model_t.apply(weights_init)
    optimizer_u = optim.Adam(model_u.parameters(), lr=1e-2)
    scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, 1000, gamma=0.8)
    optimizer_v = optim.Adam(model_v.parameters(), lr=1e-2)
    scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, 1000, gamma=0.8)
    optimizer_w = optim.Adam(model_w.parameters(), lr=1e-2)
    scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, 1000, gamma=0.8)
    optimizer_t = optim.Adam(model_t.parameters(), lr=1e-2)
    scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_t, 1000, gamma=0.8)
    print(model_u)

    # generate the data set
    xr = get_interior_points_a(50000, 4)
    xr = xr.to(device)
    xr.requires_grad_()

    best_loss, best_epoch = torch.inf, 0
    stat = time.time()
    for epoch in range(epochs+1):

        xr_x, xr_y, xr_z, xr_t = torch.chunk(xr, 4, 1)
        U = model_u(torch.cat((xr_x, xr_y, xr_z, xr_t), dim=1))
        V = model_v(torch.cat((xr_x, xr_y, xr_z, xr_t), dim=1))
        W = model_w(torch.cat((xr_x, xr_y, xr_z, xr_t), dim=1))
        T = model_t(torch.cat((xr_x, xr_y, xr_z, xr_t), dim=1))

        t_x = autograd.grad(outputs=T, inputs=xr_x,
                              grad_outputs=torch.ones_like(T),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        t_y = autograd.grad(outputs=T, inputs=xr_y,
                              grad_outputs=torch.ones_like(T),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        t_z = autograd.grad(outputs=T, inputs=xr_z,
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
        t_zz = autograd.grad(outputs=t_z, inputs=xr_z,
                              grad_outputs=torch.ones_like(t_z),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        u_x = autograd.grad(outputs=U, inputs=xr_x,
                              grad_outputs=torch.ones_like(U),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_y = autograd.grad(outputs=U, inputs=xr_y,
                              grad_outputs=torch.ones_like(U),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_w = autograd.grad(outputs=U, inputs=xr_z,
                              grad_outputs=torch.ones_like(U),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_xx = autograd.grad(outputs=u_x, inputs=xr_x,
                              grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_yy = autograd.grad(outputs=u_y, inputs=xr_y,
                              grad_outputs=torch.ones_like(u_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_zz = autograd.grad(outputs=u_w, inputs=xr_z,
                              grad_outputs=torch.ones_like(u_w),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_xy = autograd.grad(outputs=u_x, inputs=xr_y,
                              grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_xz = autograd.grad(outputs=u_x, inputs=xr_z,
                              grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        v_x = autograd.grad(outputs=V, inputs=xr_x,
                              grad_outputs=torch.ones_like(V),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_y = autograd.grad(outputs=V, inputs=xr_y,
                              grad_outputs=torch.ones_like(V),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_w = autograd.grad(outputs=V, inputs=xr_z,
                              grad_outputs=torch.ones_like(V),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_xx = autograd.grad(outputs=v_x, inputs=xr_x,
                              grad_outputs=torch.ones_like(v_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_yy = autograd.grad(outputs=v_y, inputs=xr_y,
                              grad_outputs=torch.ones_like(v_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_zz = autograd.grad(outputs=v_w, inputs=xr_z,
                              grad_outputs=torch.ones_like(v_w),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_xy = autograd.grad(outputs=v_x, inputs=xr_y,
                              grad_outputs=torch.ones_like(v_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_yz = autograd.grad(outputs=v_y, inputs=xr_z,
                              grad_outputs=torch.ones_like(v_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        w_x = autograd.grad(outputs=W, inputs=xr_x,
                              grad_outputs=torch.ones_like(W),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_y = autograd.grad(outputs=W, inputs=xr_y,
                              grad_outputs=torch.ones_like(W),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_w = autograd.grad(outputs=W, inputs=xr_z,
                              grad_outputs=torch.ones_like(W),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_xx = autograd.grad(outputs=w_x, inputs=xr_x,
                              grad_outputs=torch.ones_like(w_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_yy = autograd.grad(outputs=w_y, inputs=xr_y,
                              grad_outputs=torch.ones_like(w_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_zz = autograd.grad(outputs=w_w, inputs=xr_z,
                              grad_outputs=torch.ones_like(w_w),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_yz = autograd.grad(outputs=w_y, inputs=xr_z,
                              grad_outputs=torch.ones_like(w_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_xz = autograd.grad(outputs=w_x, inputs=xr_z,
                              grad_outputs=torch.ones_like(w_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        loss_r1 = (- (c1111 * u_xx + c1122 * v_xy + c1122 * w_xz + c1212 * u_yy + c1212 * v_xy + c1212 * u_zz + c1212 * w_xz) + b11 * t_x - f1) ** 2
        loss_r2 = (- (c1212 * u_xy + c1212 * v_xx + c1122 * u_xy + c1111 * v_yy + c1122 * w_yz + c1212 * v_zz + c1212 * w_yz) + b11 * t_y - f2) ** 2
        loss_r3 = (- (c1212 * u_xz + c1212 * w_xx + c1212 * v_yz + c1212 * w_yy + c1122 * u_xz + c1122 * v_yz + c1111 * w_zz) + b11 * t_z - f3) ** 2
        loss_r4 = (pc * t_t - a11 * (t_xx + t_yy + t_zz) - hh) ** 2
        loss_r = torch.mean(loss_r1 + loss_r2 + loss_r3 + loss_r4)
        loss = loss_r

        optimizer_u.zero_grad()
        optimizer_v.zero_grad()
        optimizer_w.zero_grad()
        optimizer_t.zero_grad()
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        optimizer_u.step()
        scheduler_u.step()
        optimizer_v.step()
        scheduler_v.step()
        optimizer_w.step()
        scheduler_w.step()
        optimizer_t.step()
        scheduler_t.step()

        if epoch % 100 == 0:

            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:', loss_r.item(),
                  'lr', optimizer_u.state_dict()['param_groups'][0]['lr'])

        if epoch > int(4 * epochs / 5):
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                torch.save(model_u.state_dict(), 'model_u0.mdl')
                torch.save(model_v.state_dict(), 'model_v0.mdl')
                torch.save(model_w.state_dict(), 'model_w0.mdl')
                torch.save(model_t.state_dict(), 'model_t0.mdl')

        if epoch % 500 == 0:

            x1 = torch.linspace(0, 1, 200)
            x2 = torch.linspace(0, 1, 200)
            X, Y = torch.meshgrid(x1, x2)
            Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None], Y.T.flatten()[:, None] * 0 + 0.5, Y.T.flatten()[:, None] * 0 + 0.1), dim=1)
            Z = Z.to(device)
            pre_u = model_u(Z)
            pre_v = model_v(Z)
            pre_w = model_w(Z)
            pre_t = model_t(Z)
            pred = pre_u.cpu().detach().numpy()
            pred_u2 = pred.reshape(200, 200)
            pred = pre_w.cpu().detach().numpy()
            pred_v2 = pred.reshape(200, 200)
            pred = pre_t.cpu().detach().numpy()
            pred_t2 = pred.reshape(200, 200)

            Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None], Y.T.flatten()[:, None] * 0 + 0.5, Y.T.flatten()[:, None] * 0 + 0.5), dim=1)
            Z = Z.to(device)
            pre_u = model_u(Z)
            pre_v = model_v(Z)
            pre_w = model_w(Z)
            pre_t = model_t(Z)
            pred = pre_u.cpu().detach().numpy()
            pred_u5 = pred.reshape(200, 200)
            pred = pre_w.cpu().detach().numpy()
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
