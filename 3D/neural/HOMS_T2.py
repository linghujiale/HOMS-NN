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
        return x * data_x * data_y * data_z * (1 - data_x) * (1 - data_y) * (1 - data_z)

def Cmatrix(E, V):
    CC = np.array([[1 / E, - V / E, - V / E, 0, 0, 0],
                   [- V / E, 1 / E, - V / E, 0, 0, 0],
                   [- V / E, - V / E, 1 / E, 0, 0, 0],
                   [0, 0, 0, 2 * (1 + V) / E, 0, 0],
                   [0, 0, 0, 0, 2 * (1 + V) / E, 0],
                   [0, 0, 0, 0, 0, 2 * (1 + V) / E]])
    CC = np.linalg.inv(CC)
    return CC


def main():
    epsilon = 1 / 5
    iniT = 1

    model_t0 = drrnnt(4, 20, 3, 1).to(device)
    model_t0.load_state_dict(torch.load('model_t0.mdl'))
    model_t0.cpu()

    in_N = 3
    m = 30
    depth = 5
    out_N = 1

    model_N1 = drrnn(in_N, m, depth, out_N).to(device)
    model_N2 = drrnn(in_N, m, depth, out_N).to(device)
    model_N3 = drrnn(in_N, m, depth, out_N).to(device)
    model_N1.load_state_dict(torch.load('model_N1.mdl'))
    model_N2.load_state_dict(torch.load('model_N2.mdl'))
    model_N3.load_state_dict(torch.load('model_N3.mdl'))
    model_N1.cpu()
    model_N2.cpu()
    model_N3.cpu()

    model_N11 = drrnn(in_N, m, depth, out_N).to(device)
    model_N12 = drrnn(in_N, m, depth, out_N).to(device)
    model_N13 = drrnn(in_N, m, depth, out_N).to(device)
    model_N11.load_state_dict(torch.load('model_N11.mdl'))
    model_N12.load_state_dict(torch.load('model_N12.mdl'))
    model_N13.load_state_dict(torch.load('model_N13.mdl'))
    model_N11.cpu()
    model_N12.cpu()
    model_N13.cpu()

    model_N21 = drrnn(in_N, m, depth, out_N).to(device)
    model_N22 = drrnn(in_N, m, depth, out_N).to(device)
    model_N23 = drrnn(in_N, m, depth, out_N).to(device)
    model_N21.load_state_dict(torch.load('model_N21.mdl'))
    model_N22.load_state_dict(torch.load('model_N22.mdl'))
    model_N23.load_state_dict(torch.load('model_N23.mdl'))
    model_N21.cpu()
    model_N22.cpu()
    model_N23.cpu()

    model_N31 = drrnn(in_N, m, depth, out_N).to(device)
    model_N32 = drrnn(in_N, m, depth, out_N).to(device)
    model_N33 = drrnn(in_N, m, depth, out_N).to(device)
    model_N31.load_state_dict(torch.load('model_N31.mdl'))
    model_N32.load_state_dict(torch.load('model_N32.mdl'))
    model_N33.load_state_dict(torch.load('model_N33.mdl'))
    model_N31.cpu()
    model_N32.cpu()
    model_N33.cpu()
    model_S = drrnn(in_N, m, 3, out_N).to(device)
    model_S.load_state_dict(torch.load('model_S.mdl'))
    model_S.cpu()
    print('load from ckpt!')

    csvt = 0.05
    csv = np.ones((125000, 1))
    for i in range(20):
        x1 = torch.linspace(0, 1, 50)
        xrx, xry, xrz = torch.meshgrid(x1, x1, x1)
        xrx = xrx.flatten()[:, None]
        xry = xry.flatten()[:, None]
        xrz = xrz.flatten()[:, None]
        xrt = xrz * 0 + csvt
        csvt = csvt + 0.05
        xrx.requires_grad_()
        xry.requires_grad_()
        xrz.requires_grad_()
        xrt.requires_grad_()
        u = model_t0(torch.cat((xry, xrx, xrz, xrt), dim=1))

        u_t = autograd.grad(outputs=u, inputs=xrt,
                            grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_x = autograd.grad(outputs=u, inputs=xrx,
                            grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_y = autograd.grad(outputs=u, inputs=xry,
                            grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_z = autograd.grad(outputs=u, inputs=xrz,
                            grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_xx = autograd.grad(outputs=u_x, inputs=xrx,
                             grad_outputs=torch.ones_like(u_x),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_yy = autograd.grad(outputs=u_y, inputs=xry,
                             grad_outputs=torch.ones_like(u_y),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_zz = autograd.grad(outputs=u_z, inputs=xrz,
                             grad_outputs=torch.ones_like(u_z),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_xy = autograd.grad(outputs=u_x, inputs=xry,
                             grad_outputs=torch.ones_like(u_x),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_yz = autograd.grad(outputs=u_y, inputs=xrz,
                             grad_outputs=torch.ones_like(u_y),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_zx = autograd.grad(outputs=u_z, inputs=xrx,
                             grad_outputs=torch.ones_like(u_z),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

        XX = xrx / epsilon - (xrx / epsilon).floor()
        YY = xry / epsilon - (xry / epsilon).floor()
        ZZ = xrz / epsilon - (xrz / epsilon).floor()

        pre_N1 = model_N1(torch.cat((XX, YY, ZZ), dim=1))
        pre_N2 = model_N2(torch.cat((XX, YY, ZZ), dim=1))
        pre_N3 = model_N3(torch.cat((XX, YY, ZZ), dim=1))

        addT1 = epsilon * (pre_N1 * u_x + pre_N2 * u_y + pre_N3 * u_z)

        pre_N11 = model_N11(torch.cat((XX, YY, ZZ), dim=1))
        pre_N12 = model_N12(torch.cat((XX, YY, ZZ), dim=1))
        pre_N13 = model_N13(torch.cat((XX, YY, ZZ), dim=1))
        pre_N21 = model_N21(torch.cat((XX, YY, ZZ), dim=1))
        pre_N22 = model_N22(torch.cat((XX, YY, ZZ), dim=1))
        pre_N23 = model_N23(torch.cat((XX, YY, ZZ), dim=1))
        pre_N31 = model_N31(torch.cat((XX, YY, ZZ), dim=1))
        pre_N32 = model_N32(torch.cat((XX, YY, ZZ), dim=1))
        pre_N33 = model_N33(torch.cat((XX, YY, ZZ), dim=1))
        pre_S = model_S(torch.cat((XX, YY, ZZ), dim=1))

        addT2 = (epsilon ** 2) * (pre_N11 * u_xx + pre_N12 * u_xy + pre_N13 * u_zx
                                  + pre_N21 * u_xy + pre_N22 * u_yy + pre_N23 * u_yz
                                  + pre_N31 * u_zx + pre_N32 * u_yz + pre_N33 * u_zz
                                  + pre_S * u_t)

        T = u + addT1 + addT2
        T = T.detach().numpy()
        csv = np.hstack((csv, T))

    np.savetxt('T2_pre.csv', csv, delimiter=',')


if __name__ == '__main__':
    main()
