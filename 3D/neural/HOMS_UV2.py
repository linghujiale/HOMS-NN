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
        data_x, data_y, data_z = torch.chunk(x, 3, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * data_z * (1 - data_x) * (1 - data_y) * (1 - data_z)

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
        data_x, data_y, data_z, data_t = torch.chunk(x, 4, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * data_z * data_t * (1 - data_x) * (1 - data_y) * (1 - data_z)

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
        data_x, data_y, data_z, data_t = torch.chunk(x, 4, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * data_z * data_t * (1 - data_x) * (1 - data_y) * (1 - data_z) + 1

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

    in_N = 4
    m = 20
    depth = 3
    out_N = 1

    model_u0 = drrnn_uv(in_N, m, depth, out_N).to(device)
    model_u0.load_state_dict(torch.load('model_u0.mdl'))
    model_u0.cpu()
    model_v0 = drrnn_uv(in_N, m, depth, out_N).to(device)
    model_v0.load_state_dict(torch.load('model_v0.mdl'))
    model_v0.cpu()
    model_w0 = drrnn_uv(in_N, m, depth, out_N).to(device)
    model_w0.load_state_dict(torch.load('model_w0.mdl'))
    model_w0.cpu()
    model_t0 = drrnn_t(in_N, m, depth, out_N).to(device)
    model_t0.load_state_dict(torch.load('model_t0.mdl'))
    model_t0.cpu()

    in_N = 3
    m = 30
    depth = 4
    out_N = 3

    model_uvw11 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw12 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw13 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw22 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw23 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw33 = drrnn(in_N, m, out_N, depth).to(device)
    model_P0 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw11.load_state_dict(torch.load('model_uvw11.mdl'))
    model_uvw12.load_state_dict(torch.load('model_uvw12.mdl'))
    model_uvw13.load_state_dict(torch.load('model_uvw31.mdl'))
    model_uvw22.load_state_dict(torch.load('model_uvw22.mdl'))
    model_uvw23.load_state_dict(torch.load('model_uvw23.mdl'))
    model_uvw33.load_state_dict(torch.load('model_uvw33.mdl'))
    model_P0.load_state_dict(torch.load('model_P0.mdl'))
    model_uvw11.cpu()
    model_uvw12.cpu()
    model_uvw13.cpu()
    model_uvw22.cpu()
    model_uvw23.cpu()
    model_uvw33.cpu()
    model_P0.cpu()
    print('low-order UC load from ckpt!')

    model_uvw111 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw111.load_state_dict(torch.load('model_uvw111.mdl'))
    model_uvw111.cpu()
    model_uvw112 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw112.load_state_dict(torch.load('model_uvw112.mdl'))
    model_uvw112.cpu()
    model_uvw113 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw113.load_state_dict(torch.load('model_uvw131.mdl'))
    model_uvw113.cpu()
    model_uvw122 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw122.load_state_dict(torch.load('model_uvw122.mdl'))
    model_uvw122.cpu()
    model_uvw123 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw123.load_state_dict(torch.load('model_uvw123.mdl'))
    model_uvw123.cpu()
    model_uvw133 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw133.load_state_dict(torch.load('model_uvw133.mdl'))
    model_uvw133.cpu()

    model_uvw211 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw211.load_state_dict(torch.load('model_uvw211.mdl'))
    model_uvw211.cpu()
    model_uvw212 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw212.load_state_dict(torch.load('model_uvw212.mdl'))
    model_uvw212.cpu()
    model_uvw213 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw213.load_state_dict(torch.load('model_uvw231.mdl'))
    model_uvw213.cpu()
    model_uvw222 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw222.load_state_dict(torch.load('model_uvw222.mdl'))
    model_uvw222.cpu()
    model_uvw223 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw223.load_state_dict(torch.load('model_uvw223.mdl'))
    model_uvw223.cpu()
    model_uvw233 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw233.load_state_dict(torch.load('model_uvw233.mdl'))
    model_uvw233.cpu()

    model_uvw311 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw311.load_state_dict(torch.load('model_uvw311.mdl'))
    model_uvw311.cpu()
    model_uvw312 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw312.load_state_dict(torch.load('model_uvw312.mdl'))
    model_uvw312.cpu()
    model_uvw313 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw313.load_state_dict(torch.load('model_uvw331.mdl'))
    model_uvw313.cpu()
    model_uvw322 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw322.load_state_dict(torch.load('model_uvw322.mdl'))
    model_uvw322.cpu()
    model_uvw323 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw323.load_state_dict(torch.load('model_uvw323.mdl'))
    model_uvw323.cpu()
    model_uvw333 = drrnn(in_N, m, out_N, depth).to(device)
    model_uvw333.load_state_dict(torch.load('model_uvw333.mdl'))
    model_uvw333.cpu()

    model_P1 = drrnn(in_N, m, out_N, depth).to(device)
    model_P1.load_state_dict(torch.load('model_P1.mdl'))
    model_P1.cpu()
    model_P2 = drrnn(in_N, m, out_N, depth).to(device)
    model_P2.load_state_dict(torch.load('model_P2.mdl'))
    model_P2.cpu()
    model_P3 = drrnn(in_N, m, out_N, depth).to(device)
    model_P3.load_state_dict(torch.load('model_P3.mdl'))
    model_P3.cpu()
    print('high-order UC load from ckpt!')

    csvt = 0.05
    csvu = np.ones((125000, 1))
    csvv = np.ones((125000, 1))
    csvw = np.ones((125000, 1))

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
        zz = torch.cat((xrx, xry, xrz, xrt), dim=1)

        u = model_u0(zz)
        v = model_v0(zz)
        w = model_w0(zz)
        t = model_t0(zz)

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
        # v
        v_x = autograd.grad(outputs=v, inputs=xrx,
                            grad_outputs=torch.ones_like(v),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_y = autograd.grad(outputs=v, inputs=xry,
                            grad_outputs=torch.ones_like(v),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_z = autograd.grad(outputs=v, inputs=xrz,
                            grad_outputs=torch.ones_like(v),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_xx = autograd.grad(outputs=v_x, inputs=xrx,
                             grad_outputs=torch.ones_like(v_x),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_yy = autograd.grad(outputs=v_y, inputs=xry,
                             grad_outputs=torch.ones_like(v_y),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_zz = autograd.grad(outputs=v_z, inputs=xrz,
                             grad_outputs=torch.ones_like(v_z),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_xy = autograd.grad(outputs=v_x, inputs=xry,
                             grad_outputs=torch.ones_like(v_x),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_yz = autograd.grad(outputs=v_y, inputs=xrz,
                             grad_outputs=torch.ones_like(v_y),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_zx = autograd.grad(outputs=v_z, inputs=xrx,
                             grad_outputs=torch.ones_like(v_z),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        # w
        w_x = autograd.grad(outputs=w, inputs=xrx,
                            grad_outputs=torch.ones_like(w),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_y = autograd.grad(outputs=w, inputs=xry,
                            grad_outputs=torch.ones_like(w),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_z = autograd.grad(outputs=w, inputs=xrz,
                            grad_outputs=torch.ones_like(w),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_xx = autograd.grad(outputs=w_x, inputs=xrx,
                             grad_outputs=torch.ones_like(w_x),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_yy = autograd.grad(outputs=w_y, inputs=xry,
                             grad_outputs=torch.ones_like(w_y),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_zz = autograd.grad(outputs=w_z, inputs=xrz,
                             grad_outputs=torch.ones_like(w_z),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_xy = autograd.grad(outputs=w_x, inputs=xry,
                             grad_outputs=torch.ones_like(w_x),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_yz = autograd.grad(outputs=w_y, inputs=xrz,
                             grad_outputs=torch.ones_like(w_y),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_zx = autograd.grad(outputs=w_z, inputs=xrx,
                             grad_outputs=torch.ones_like(w_z),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        # t
        t_x = autograd.grad(outputs=t, inputs=xrx,
                            grad_outputs=torch.ones_like(t),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        t_y = autograd.grad(outputs=t, inputs=xry,
                            grad_outputs=torch.ones_like(t),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        t_z = autograd.grad(outputs=t, inputs=xrz,
                            grad_outputs=torch.ones_like(t),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        XX = xrx / epsilon - (xrx / epsilon).floor()
        YY = xry / epsilon - (xry / epsilon).floor()
        ZZ = xrz / epsilon - (xrz / epsilon).floor()
        XYZ = torch.cat((XX, YY, ZZ), dim=1)

        pre_uvw11 = model_uvw11(XYZ)
        pre_u11, pre_v11, pre_w11 = torch.chunk(pre_uvw11, 3, 1)
        pre_uvw12 = model_uvw12(XYZ)
        pre_u12, pre_v12, pre_w12 = torch.chunk(pre_uvw12, 3, 1)
        pre_uvw13 = model_uvw13(XYZ)
        pre_u13, pre_v13, pre_w13 = torch.chunk(pre_uvw13, 3, 1)
        pre_uvw22 = model_uvw22(XYZ)
        pre_u22, pre_v22, pre_w22 = torch.chunk(pre_uvw22, 3, 1)
        pre_uvw23 = model_uvw23(XYZ)
        pre_u23, pre_v23, pre_w23 = torch.chunk(pre_uvw23, 3, 1)
        pre_uvw33 = model_uvw33(XYZ)
        pre_u33, pre_v33, pre_w33 = torch.chunk(pre_uvw33, 3, 1)
        pre_P0 = model_P0(XYZ) / 100
        pre_P01, pre_P02, pre_P03 = torch.chunk(pre_P0, 3, 1)

        addU1 = epsilon * (pre_u11 * u_x + pre_u12 * (u_y + v_x) + pre_u13 * (u_z + w_x)
                           + pre_u22 * v_y + pre_u23 * (v_z + w_y) + pre_u33 * w_z - pre_P01 * (t - iniT))
        addV1 = epsilon * (pre_v11 * u_x + pre_v12 * (u_y + v_x) + pre_v13 * (u_z + w_x)
                           + pre_v22 * v_y + pre_v23 * (v_z + w_y) + pre_v33 * w_z - pre_P02 * (t - iniT))
        addW1 = epsilon * (pre_w11 * u_x + pre_w12 * (u_y + v_x) + pre_w13 * (u_z + w_x)
                           + pre_w22 * v_y + pre_w23 * (v_z + w_y) + pre_w33 * w_z - pre_P03 * (t - iniT))

        del pre_uvw11, pre_uvw12, pre_uvw13, pre_uvw22, pre_uvw23, pre_uvw33
        del pre_u11, pre_u12, pre_u13, pre_u22, pre_u23, pre_u33
        del pre_v11, pre_v12, pre_v13, pre_v22, pre_v23, pre_v33
        del pre_w11, pre_w12, pre_w13, pre_w22, pre_w23, pre_w33

        pre_uvw111 = model_uvw111(XYZ)
        pre_u111, pre_v111, pre_w111 = torch.chunk(pre_uvw111, 3, 1)
        pre_uvw112 = model_uvw112(XYZ)
        pre_u112, pre_v112, pre_w112 = torch.chunk(pre_uvw112, 3, 1)
        pre_uvw113 = model_uvw113(XYZ)
        pre_u113, pre_v113, pre_w113 = torch.chunk(pre_uvw113, 3, 1)
        pre_uvw122 = model_uvw122(XYZ)
        pre_u122, pre_v122, pre_w122 = torch.chunk(pre_uvw122, 3, 1)
        pre_uvw123 = model_uvw123(XYZ)
        pre_u123, pre_v123, pre_w123 = torch.chunk(pre_uvw123, 3, 1)
        pre_uvw133 = model_uvw133(XYZ)
        pre_u133, pre_v133, pre_w133 = torch.chunk(pre_uvw133, 3, 1)

        pre_uvw211 = model_uvw211(XYZ)
        pre_u211, pre_v211, pre_w211 = torch.chunk(pre_uvw211, 3, 1)
        pre_uvw212 = model_uvw212(XYZ)
        pre_u212, pre_v212, pre_w212 = torch.chunk(pre_uvw212, 3, 1)
        pre_uvw213 = model_uvw213(XYZ)
        pre_u213, pre_v213, pre_w213 = torch.chunk(pre_uvw213, 3, 1)
        pre_uvw222 = model_uvw222(XYZ)
        pre_u222, pre_v222, pre_w222 = torch.chunk(pre_uvw222, 3, 1)
        pre_uvw223 = model_uvw223(XYZ)
        pre_u223, pre_v223, pre_w223 = torch.chunk(pre_uvw223, 3, 1)
        pre_uvw233 = model_uvw233(XYZ)
        pre_u233, pre_v233, pre_w233 = torch.chunk(pre_uvw233, 3, 1)

        pre_uvw311 = model_uvw311(XYZ)
        pre_u311, pre_v311, pre_w311 = torch.chunk(pre_uvw311, 3, 1)
        pre_uvw312 = model_uvw312(XYZ)
        pre_u312, pre_v312, pre_w312 = torch.chunk(pre_uvw312, 3, 1)
        pre_uvw313 = model_uvw313(XYZ)
        pre_u313, pre_v313, pre_w313 = torch.chunk(pre_uvw313, 3, 1)
        pre_uvw322 = model_uvw322(XYZ)
        pre_u322, pre_v322, pre_w322 = torch.chunk(pre_uvw322, 3, 1)
        pre_uvw323 = model_uvw323(XYZ)
        pre_u323, pre_v323, pre_w323 = torch.chunk(pre_uvw323, 3, 1)
        pre_uvw333 = model_uvw333(XYZ)
        pre_u333, pre_v333, pre_w333 = torch.chunk(pre_uvw333, 3, 1)

        pre_P1 = model_P1(XYZ) / 100
        pre_P11, pre_P12, pre_P13 = torch.chunk(pre_P1, 3, 1)
        pre_P2 = model_P2(XYZ) / 100
        pre_P21, pre_P22, pre_P23 = torch.chunk(pre_P2, 3, 1)
        pre_P3 = model_P3(XYZ) / 100
        pre_P31, pre_P32, pre_P33 = torch.chunk(pre_P3, 3, 1)

        addU2 = (epsilon ** 2) * (pre_u111 * u_xx + pre_u112 * (u_xy + v_xx) + pre_u113 * (u_zx + w_xx) + pre_u122 * v_xy +
                                  pre_u123 * (v_zx + w_xy) + pre_u133 * w_zx + pre_u211 * u_xy + pre_u212 * (u_yy + v_xy) +
                                  pre_u213 * (u_yz + w_xy) + pre_u222 * v_yy + pre_u223 * (v_yz + w_yy) + pre_u233 * w_yz +
                                  pre_u311 * u_zx + pre_u312 * (u_yz + v_zx) + pre_u313 * (u_zz + w_zx) + pre_u322 * v_yz +
                                  pre_u323 * (v_zz + w_yz) + pre_u333 * w_zz - pre_P11 * t_x - pre_P21 * t_y - pre_P31 * t_z)

        addV2 = (epsilon ** 2) * (pre_v111 * u_xx + pre_v112 * (u_xy + v_xx) + pre_v113 * (u_zx + w_xx) + pre_v122 * v_xy +
                                  pre_v123 * (v_zx + w_xy) + pre_v133 * w_zx + pre_v211 * u_xy + pre_v212 * (u_yy + v_xy) +
                                  pre_v213 * (u_yz + w_xy) + pre_v222 * v_yy + pre_v223 * (v_yz + w_yy) + pre_v233 * w_yz +
                                  pre_v311 * u_zx + pre_v312 * (u_yz + v_zx) + pre_v313 * (u_zz + w_zx) + pre_v322 * v_yz +
                                  pre_v323 * (v_zz + w_yz) + pre_v333 * w_zz - pre_P12 * t_x - pre_P22 * t_y - pre_P32 * t_z)

        addW2 = (epsilon ** 2) * (pre_w111 * u_xx + pre_w112 * (u_xy + v_xx) + pre_w113 * (u_zx + w_xx) + pre_w122 * v_xy +
                                  pre_w123 * (v_zx + w_xy) + pre_w133 * w_zx + pre_w211 * u_xy + pre_w212 * (u_yy + v_xy) +
                                  pre_w213 * (u_yz + w_xy) + pre_w222 * v_yy + pre_w223 * (v_yz + w_yy) + pre_w233 * w_yz +
                                  pre_w311 * u_zx + pre_w312 * (u_yz + v_zx) + pre_w313 * (u_zz + w_zx) + pre_w322 * v_yz +
                                  pre_w323 * (v_zz + w_yz) + pre_w333 * w_zz - pre_P13 * t_x - pre_P23 * t_y - pre_P33 * t_z)

        U = u + addU1 + addU2
        V = v + addV1 + addV2
        W = w + addW1 + addW2

        U = U.detach().numpy()
        V = V.detach().numpy()
        W = W.detach().numpy()

        csvu = np.hstack((csvu, U))
        csvv = np.hstack((csvv, V))
        csvw = np.hstack((csvw, W))

    np.savetxt('U2_pre.csv', csvu, delimiter=',')
    np.savetxt('V2_pre.csv', csvv, delimiter=',')
    np.savetxt('W2_pre.csv', csvw, delimiter=',')


if __name__ == '__main__':
    main()
