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
sys.stdout = Logger('P1.txt')

print(path)
print(os.path.dirname(__file__))
print('------------------')
for i in range(5, 10):
    print("this is the %d times" % i)

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
        return x * data_x * data_y * data_z * (1 - data_x) * (1 - data_y) * (1 - data_z)

def get_interior_points_a(N,d):

    return torch.rand(N,d)

def get_interior_points_c(N):

    x1 = torch.linspace(0, 1, N)
    X, Y, Z = torch.meshgrid(x1, x1, x1)
    Z = torch.cat((Y.flatten()[:, None],
                   X.flatten()[:, None],
                   Z.flatten()[:, None]
                   ), dim=1)
    return Z

def weights_init(m):

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def func_thermal(X, conductivity1, conductivity2, func):
    # conductivity1 is out thermal conductivity conductivity2 is inner
    k = 100000  # k 越大，间断越小
    r = 0.3  # 圆半径
    x, y, z = torch.chunk(X, 3, 1)
    if func == "conductivity":
        output = (1 / (1 + torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - r ** 2)))) \
                 * (conductivity1 - conductivity2) + conductivity2
    if func == "partial_x":
        output = (conductivity1 - conductivity2) * (
                (2 * k * (x - 0.5) * torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - r ** 2)))
                / ((1 + torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - r ** 2))) ** 2))
    if func == "partial_y":
        output = (conductivity1 - conductivity2) * (
                (2 * k * (y - 0.5) * torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - r ** 2)))
                / ((1 + torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - r ** 2))) ** 2))
    if func == "partial_z":
        output = (conductivity1 - conductivity2) * (
                (2 * k * (z - 0.5) * torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - r ** 2)))
                / ((1 + torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - r ** 2))) ** 2))
    return output

def Cmatrix(E, V):
    CC = np.array([[1 / E, - V / E, - V / E, 0, 0, 0],
                   [- V / E, 1 / E, - V / E, 0, 0, 0],
                   [- V / E, - V / E, 1 / E, 0, 0, 0],
                   [0, 0, 0, 2 * (1 + V) / E, 0, 0],
                   [0, 0, 0, 0, 2 * (1 + V) / E, 0],
                   [0, 0, 0, 0, 0, 2 * (1 + V) / E]])
    CC = np.linalg.inv(CC)
    return CC
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

    thermal1 = 100
    thermal2 = 1
    beta1 = 50
    beta2 = 5
    E1 = 300
    E2 = 6
    v1 = 0.25
    v2 = 0.25
    pc1 = 50
    pc2 = 5
    epochs = 10000

    in_N = 3
    m = 30
    out_N = 3
    depth = 4

    data = pd.read_csv(r"E:\pycharm\project6\3d\data\P11.csv")
    data = np.array(data)
    data = data[0, 0:27000]
    datau = torch.tensor(data).to(device)
    print("max data: ", format(torch.max(datau), '.5f'), " max data: ", format(torch.min(datau), '.5f'))

    data = pd.read_csv(r"E:\pycharm\project6\3d\data\P12.csv")
    data = np.array(data)
    data = data[0, 0:27000]
    datav = torch.tensor(data).to(device)
    print("max data: ", format(torch.max(datav), '.5f'), " max data: ", format(torch.min(datav), '.5f'))

    data = pd.read_csv(r"E:\pycharm\project6\3d\data\P13.csv")
    data = np.array(data)
    data = data[0, 0:27000]
    dataw = torch.tensor(data).to(device)
    print("max data: ", format(torch.max(dataw), '.5f'), " max data: ", format(torch.min(dataw), '.5f'))

    model = drrnn(in_N, m, depth, out_N).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.9)
    print(model)

    CC1 = Cmatrix(E1, v1)
    CC2 = Cmatrix(E2, v2)
    print("C1: ", CC1)
    print("C2: ", CC2)

    # load lower-order UC
    model_P0 = drrnn(in_N, m, depth, out_N).to(device)
    model_P0.load_state_dict(torch.load('model_P0.mdl'))
    model_N1 = drrnn(in_N, m, 5, 1).to(device)
    model_N1.load_state_dict(torch.load('model_N1.mdl'))
    print('load from ckpt!')

    # calculate C0
    xr_c = get_interior_points_c(50)
    C1 = func_thermal(xr_c, CC1[0, 0], CC2[0, 0], "conductivity").to(device) * 100
    C2 = func_thermal(xr_c, CC1[0, 1], CC2[0, 1], "conductivity").to(device) * 100
    beta = func_thermal(xr_c, beta1, beta2, "conductivity").to(device)
    xr_c = xr_c.to(device)
    xr_c.requires_grad_()
    output = model_P0(xr_c) / 100
    output_u, output_v, output_w = torch.chunk(output, 3, 1)

    grads_u = autograd.grad(outputs=output_u, inputs=xr_c,
                            grad_outputs=torch.ones_like(output_u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_x, u_y, u_z = torch.chunk(grads_u, 3, 1)
    grads_v = autograd.grad(outputs=output_v, inputs=xr_c,
                            grad_outputs=torch.ones_like(output_v),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    v_x, v_y, v_z = torch.chunk(grads_v, 3, 1)
    grads_w = autograd.grad(outputs=output_w, inputs=xr_c,
                            grad_outputs=torch.ones_like(output_w),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    w_x, w_y, w_z = torch.chunk(grads_w, 3, 1)

    b0 = torch.mean(beta + C1 * u_x + C2 * v_y + C2 * w_z)
    print("b0: ", b0)

    # if epoch % 5 == 0:
    # generate the data set
    xr = quasirandom(200000, "sobol")
    xr = xr.to(device)
    C1 = func_thermal(xr, CC1[0, 0], CC2[0, 0], "conductivity")
    C2 = func_thermal(xr, CC1[1, 0], CC2[1, 0], "conductivity")
    C3 = func_thermal(xr, CC1[3, 3], CC2[3, 3], "conductivity")
    beta = func_thermal(xr, beta1, beta2, "conductivity")
    xr.requires_grad_()
    xrx, xry, xrz = torch.chunk(xr, 3, 1)

    best_loss, best_epoch = 100010, 0
    stat = time.time()
    for epoch in range(epochs+1):

        uvw = model(torch.cat([xrx, xry, xrz], dim=1))
        u, v, w = torch.chunk(uvw, 3, 1)
        grads_u = autograd.grad(outputs=u, inputs=xr,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_x, u_y, u_z = torch.chunk(grads_u, 3, 1)
        grads_v = autograd.grad(outputs=v, inputs=xr,
                              grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_x, v_y, v_z = torch.chunk(grads_v, 3, 1)
        grads_w = autograd.grad(outputs=w, inputs=xr,
                              grad_outputs=torch.ones_like(w),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        w_x, w_y, w_z = torch.chunk(grads_w, 3, 1)

        uvw = model_P0(xr) / 100
        lu, lv, lw = torch.chunk(uvw, 3, 1)
        grads_lu = autograd.grad(outputs=lu, inputs=xr,
                              grad_outputs=torch.ones_like(lu),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        lu_x, lu_y, lu_z = torch.chunk(grads_lu, 3, 1)
        grads_lv = autograd.grad(outputs=lv, inputs=xr,
                              grad_outputs=torch.ones_like(lv),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        lv_x, lv_y, lv_z = torch.chunk(grads_lv, 3, 1)
        grads_lw = autograd.grad(outputs=lw, inputs=xr,
                              grad_outputs=torch.ones_like(lw),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        lw_x, lw_y, lw_z = torch.chunk(grads_lw, 3, 1)

        ln = model_N1(xr)

        loss_r1 = 0.5 * (torch.squeeze(C1 * u_x * u_x) + torch.squeeze(C2 * v_y * u_x) + torch.squeeze(C2 * u_x * w_z)
                         + torch.squeeze(C3 * u_y * u_y) + torch.squeeze(C3 * v_x * u_y) + torch.squeeze(C3 * u_z * u_z) + torch.squeeze(C3 * w_x * u_z)
                         + torch.squeeze(C3 * u_y * v_x) + torch.squeeze(C3 * v_x * v_x) + torch.squeeze(C3 * v_z * v_z) + torch.squeeze(C3 * w_y * v_z)
                         + torch.squeeze(C2 * u_x * v_y) + torch.squeeze(C1 * v_y * v_y) + torch.squeeze(C2 * w_z * v_y)
                         + torch.squeeze(C3 * u_z * w_x) + torch.squeeze(C3 * w_x * w_x) + torch.squeeze(C3 * w_y * w_y) + torch.squeeze(C3 * w_y * v_z)
                         + torch.squeeze(C2 * u_x * w_z) + torch.squeeze(C2 * w_z * v_y) + torch.squeeze(C1 * w_z * w_z))
        loss_r2 = torch.squeeze((beta - b0) * u)
        loss_r3 = torch.squeeze((C1 * lu_x + C2 * lv_y + C2 * lw_z) * u)
        loss_r4 = torch.squeeze((C3 * lu_y + C3 * lv_x) * v)
        loss_r5 = torch.squeeze((C3 * lu_z + C3 * lw_x) * w)
        loss_r6 = torch.squeeze((C1 * u_x + C2 * v_y + C2 * w_z) * lu)
        loss_r7 = torch.squeeze((C3 * u_y + C3 * v_x) * lv)
        loss_r8 = torch.squeeze((C3 * u_z + C3 * w_x) * lw)
        loss_r9 = torch.squeeze((u_x + v_y + w_z) * beta * ln)
        loss_r = torch.mean(loss_r1 - loss_r2 - loss_r3 - loss_r4 - loss_r5 + loss_r6 + loss_r7 + loss_r8 + loss_r9)
        loss = loss_r

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            x1 = torch.linspace(0, 1, 30)
            X, Y, Z = torch.meshgrid(x1, x1, x1)
            Z = torch.cat((Y.flatten()[:, None],
                           X.flatten()[:, None],
                           Z.flatten()[:, None]
                           ), dim=1)
            Z = Z.to(device)
            pred = model(Z) / 100
            pred_u, pred_v, pred_w = torch.chunk(pred, 3, 1)
            pred_u = torch.squeeze(pred_u)
            pred_v = torch.squeeze(pred_v)
            pred_w = torch.squeeze(pred_w)
            pred = datau - pred_u
            UL2 = torch.norm(pred, p=2) / torch.norm(datau, p=2)
            pred = datav - pred_v
            VL2 = torch.norm(pred, p=2) / torch.norm(datav, p=2)
            pred = dataw - pred_w
            WL2 = torch.norm(pred, p=2) / torch.norm(dataw, p=2)
            print('epoch:', epoch, 'loss:', format(loss.item(), '.5f'), 'loss_r:', 'lr',
                  format(optimizer.state_dict()['param_groups'][0]['lr'], '.8f'),
                  "UL2Rerror: ", format(UL2.item(), '.5f'), "VL2Rerror: ", format(VL2.item(), '.5f'), "WL2Rerror: ",
                  format(WL2.item(), '.5f'))

            if epoch % 1000 == 0:
                print('maxU: ', format(torch.max(pred_u).item(), '.5f'), 'minU: ',
                      format(torch.min(pred_u).item(), '.5f'),
                      ' maxV: ', format(torch.max(pred_v).item(), '.5f'), ' minV: ',
                      format(torch.min(pred_v).item(), '.5f'),
                      ' maxW: ', format(torch.max(pred_w).item(), '.5f'), ' minW: ',
                      format(torch.min(pred_w).item(), '.5f'))

    torch.save(model.state_dict(), 'model_P1.mdl')
    print("Running time: ", time.time()-stat)

if __name__ == '__main__':
    main()
