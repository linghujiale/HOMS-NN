import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
import numpy as np
import time

from matplotlib.pyplot import MultipleLocator

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

    def __init__(self, in_N, m, out_N, depth):
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
        data_x, data_y = torch.chunk(x, 2, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * (1 - data_x) * (1 - data_y)

class drrnn_uv(nn.Module):

    def __init__(self, in_N, m, depth, out_N):
        super(drrnn_uv, self).__init__()
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

def Cmatrix(E, V):
    CC = np.array([[E / (1 - V ** 2), (E * V) / (1 - V ** 2), 0],
                   [(E * V) / (1 - V ** 2), E / (1 - V ** 2), 0],
                   [0, 0, E / (2 * (1 + V))]])
    return CC

def main():

    E1 = 10
    E2 = 1
    v1 = 0.25
    v2 = 0.25
    iniT = 1
    epochs = 30000
    f = 10

    epsilon = 1 / 10

    C1 = Cmatrix(E1, v1)
    C2 = Cmatrix(E2, v2)
    print("C1: ", C1)
    print("C2: ", C2)

    model_u0 = drrnn_uv(3, 20, 3, 1).to(device)
    model_u0.load_state_dict(torch.load('model_u0.mdl'))
    model_u0.cpu()
    model_v0 = drrnn_uv(3, 20, 3, 1).to(device)
    model_v0.load_state_dict(torch.load('model_v0.mdl'))
    model_v0.cpu()
    model_t0 = drrnn_t(3, 20, 3, 1).to(device)
    model_t0.load_state_dict(torch.load('model_t0.mdl'))
    model_t0.cpu()

    in_N = 2
    m = 30
    depth = 4
    out_N = 2

    model_uv11 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv22 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv12 = drrnn(in_N, m, out_N, depth).to(device)
    model_P0 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv11.load_state_dict(torch.load('model_uv11.mdl'))
    model_uv22.load_state_dict(torch.load('model_uv22.mdl'))
    model_uv12.load_state_dict(torch.load('model_uv12.mdl'))
    model_P0.load_state_dict(torch.load('model_P0.mdl'))
    model_uv11.cpu()
    model_uv22.cpu()
    model_uv12.cpu()
    model_P0.cpu()

    model_uv111 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv112 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv122 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv211 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv212 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv222 = drrnn(in_N, m, out_N, depth).to(device)
    model_P1 = drrnn(in_N, m, out_N, depth).to(device)
    model_P2 = drrnn(in_N, m, out_N, depth).to(device)
    model_uv111.load_state_dict(torch.load('model_uv111.mdl'))
    model_uv112.load_state_dict(torch.load('model_uv112.mdl'))
    model_uv122.load_state_dict(torch.load('model_uv122.mdl'))
    model_uv211.load_state_dict(torch.load('model_uv211.mdl'))
    model_uv212.load_state_dict(torch.load('model_uv212.mdl'))
    model_uv222.load_state_dict(torch.load('model_uv222.mdl'))
    model_P1.load_state_dict(torch.load('model_P1.mdl'))
    model_P2.load_state_dict(torch.load('model_P2.mdl'))
    model_uv111.cpu()
    model_uv112.cpu()
    model_uv122.cpu()
    model_uv211.cpu()
    model_uv212.cpu()
    model_uv222.cpu()
    model_P1.cpu()
    model_P2.cpu()

    print('load from ckpt!')

    x1 = torch.linspace(0, 1, 500)
    x2 = torch.linspace(0, 1, 500)
    X, Y = torch.meshgrid(x1, x2)
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]

    XX = X / epsilon - (X / epsilon).floor()
    YY = Y / epsilon - (Y / epsilon).floor()

    pre_uv11 = model_uv11(torch.cat((XX, YY), dim=1))
    pre_uv12 = model_uv12(torch.cat((XX, YY), dim=1))
    pre_uv22 = model_uv22(torch.cat((XX, YY), dim=1))
    pre_P0 = model_P0(torch.cat((XX, YY), dim=1)) / 100
    pre_u11, pre_v11 = torch.chunk(pre_uv11, 2, 1)
    pre_u12, pre_v12 = torch.chunk(pre_uv12, 2, 1)
    pre_u22, pre_v22 = torch.chunk(pre_uv22, 2, 1)
    pre_p01, pre_p02 = torch.chunk(pre_P0, 2, 1)

    pre_uv111 = model_uv111(torch.cat((XX, YY), dim=1))
    pre_uv112 = model_uv112(torch.cat((XX, YY), dim=1))
    pre_uv122 = model_uv122(torch.cat((XX, YY), dim=1))
    pre_uv211 = model_uv211(torch.cat((XX, YY), dim=1))
    pre_uv212 = model_uv212(torch.cat((XX, YY), dim=1))
    pre_uv222 = model_uv222(torch.cat((XX, YY), dim=1))
    pre_P1 = model_P1(torch.cat((XX, YY), dim=1)) / 100
    pre_P2 = model_P2(torch.cat((XX, YY), dim=1)) / 100

    pre_u111, pre_v111 = torch.chunk(pre_uv111, 2, 1)
    pre_u112, pre_v112 = torch.chunk(pre_uv112, 2, 1)
    pre_u122, pre_v122 = torch.chunk(pre_uv122, 2, 1)
    pre_u211, pre_v211 = torch.chunk(pre_uv211, 2, 1)
    pre_u212, pre_v212 = torch.chunk(pre_uv212, 2, 1)
    pre_u222, pre_v222 = torch.chunk(pre_uv222, 2, 1)
    pre_P11, pre_P12 = torch.chunk(pre_P1, 2, 1)
    pre_P21, pre_P22 = torch.chunk(pre_P2, 2, 1)

    csvt = 0.05
    csv = np.ones((250000, 1))
    csu = np.ones((250000, 1))
    for i in range(20):

        Z = X * 0 + csvt
        csvt = csvt + 0.05
        X.requires_grad_()
        Y.requires_grad_()
        Z.requires_grad_()
        pre_u0 = model_u0(torch.cat((X, Y, Z), dim=1))
        pre_v0 = model_v0(torch.cat((X, Y, Z), dim=1))
        pre_t0 = model_t0(torch.cat((X, Y, Z), dim=1))

        t0_x = autograd.grad(outputs=pre_t0, inputs=X,
                             grad_outputs=torch.ones_like(pre_t0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        t0_y = autograd.grad(outputs=pre_t0, inputs=Y,
                             grad_outputs=torch.ones_like(pre_t0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

        u0_x = autograd.grad(outputs=pre_u0, inputs=X,
                             grad_outputs=torch.ones_like(pre_u0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        u0_y = autograd.grad(outputs=pre_u0, inputs=Y,
                             grad_outputs=torch.ones_like(pre_u0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        u0_xx = autograd.grad(outputs=u0_x, inputs=X,
                              grad_outputs=torch.ones_like(u0_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u0_xy = autograd.grad(outputs=u0_x, inputs=Y,
                              grad_outputs=torch.ones_like(u0_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u0_yy = autograd.grad(outputs=u0_y, inputs=Y,
                              grad_outputs=torch.ones_like(u0_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        v0_x = autograd.grad(outputs=pre_v0, inputs=X,
                             grad_outputs=torch.ones_like(pre_v0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        v0_y = autograd.grad(outputs=pre_v0, inputs=Y,
                             grad_outputs=torch.ones_like(pre_v0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        v0_xx = autograd.grad(outputs=v0_x, inputs=X,
                              grad_outputs=torch.ones_like(v0_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v0_xy = autograd.grad(outputs=v0_x, inputs=Y,
                              grad_outputs=torch.ones_like(v0_x),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v0_yy = autograd.grad(outputs=v0_y, inputs=Y,
                              grad_outputs=torch.ones_like(v0_y),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        V1 = pre_v0 + epsilon * (pre_v11 * u0_x + pre_v22 * v0_y + pre_v12 * (u0_y + v0_x) - pre_p02 * (pre_t0 - iniT))

        V2 = V1 + (epsilon ** 2) * (pre_v111 * u0_xx + pre_v112 * (u0_xy + v0_xx) + pre_v122 * v0_xy + pre_v211 * u0_xy
                                    + pre_v212 * (u0_yy + v0_xy) + pre_v222 * v0_yy - pre_P12 * t0_x - pre_P22 * t0_y)

        U1 = pre_u0 + epsilon * (pre_u11 * u0_x + pre_u22 * v0_y + pre_u12 * (u0_y + v0_x) - pre_p01 * (pre_t0 - iniT))

        U2 = U1 + (epsilon ** 2) * (pre_u111 * u0_xx + pre_u112 * (u0_xy + v0_xx) + pre_u122 * v0_xy + pre_u211 * u0_xy
                                    + pre_u212 * (u0_yy + v0_xy) + pre_u222 * v0_yy - pre_P11 * t0_x - pre_P21 * t0_y)
        V0 = pre_v0.detach().numpy()
        V2 = V2.detach().numpy()
        U0 = pre_u0.detach().numpy()
        U2 = U2.detach().numpy()

        csu = np.hstack((csu, U2))
        csv = np.hstack((csv, V2))
        print("time: ", csvt)

    np.savetxt('V2_pre.csv', csv, delimiter=',')
    np.savetxt('U2_pre.csv', csu, delimiter=',')

if __name__ == '__main__':
    main()
