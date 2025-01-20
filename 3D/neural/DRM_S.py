import torch
import torch.nn as nn
from torch import optim, autograd
from collections import OrderedDict
import numpy as np
import time
import pandas as pd
import os
import sys
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
sys.stdout = Logger('S.txt')

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

def get_interior_points_a(N, d):

    return torch.rand(N, d)

def get_interior_points_c(N):

    x1 = torch.linspace(0, 1, N)
    X, Y, Z = torch.meshgrid(x1, x1, x1)
    Z = torch.cat((Y.flatten()[:, None],
                   X.flatten()[:, None],
                   Z.flatten()[:, None]
                   ), dim=1)
    return Z

def get_boundary_points(N):
    index = torch.rand(N, 1)
    # index1 = torch.rand(N,1) * 2 - 1
    index1 = torch.rand(N, 1)
    xb1 = torch.cat((index, torch.zeros_like(index)), dim=1)
    xb2 = torch.cat((index1, torch.ones_like(index1)), dim=1)
    xb3 = torch.cat((index1, torch.full_like(index1, 0)), dim=1)
    xb4 = torch.cat((torch.ones_like(index1), index1), dim=1)
    xb5 = torch.cat((torch.full_like(index1, 0), index1), dim=1)
    xb = torch.cat((xb2, xb3, xb4, xb5), dim=0)

    return xb

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def func_thermal(X, conductivity1, conductivity2, func):
    # conductivity1 is out thermal conductivity conductivity2 is inner
    k = 30000  # k 越大，间断越小
    r = 0.3  # 圆半径
    x, y, z = torch.chunk(X, 3, 1)
    if func == "conductivity":
        output = (1 / (1 + torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 - r ** 2)))) \
                 * (conductivity1 - conductivity2) + conductivity2
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
    rho1 = 45
    rho2 = 4.5
    pc1 = 50
    pc2 = 5
    epochs = 10000

    in_N = 3
    m = 30
    depth = 3
    out_N = 1

    data = pd.read_csv(r"E:\pycharm\project6\3d\data\S.csv")
    data = np.array(data)
    data = data[0, 0:27000]
    datau = torch.tensor(data).to(device)
    print("max data: ", format(torch.max(datau), '.5f'), " max data: ", format(torch.min(datau), '.5f'))

    # calculate C0
    xr_c = get_interior_points_c(50)
    pc = func_thermal(xr_c, pc1, pc2, "conductivity")
    pc0 = torch.mean(pc).to(device)
    print("pc0", pc0)

    model = drrnn(in_N, m, out_N, depth).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.95)
    print(model)

    xr = quasirandom(300000, "halton")
    xr = xr.to(device)
    thermalxr = func_thermal(xr, thermal1, thermal2, "conductivity")
    pc = func_thermal(xr, pc1, pc2, "conductivity")
    xr.requires_grad_()
    # generate the data set
    best_loss, best_epoch = 10000, 0
    start = time.time()
    for epoch in range(epochs + 1):

        output_r = model(xr)
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss_1 = 0.5 * torch.sum(thermalxr * torch.pow(grads, 2), dim=1)
        loss_2 = torch.squeeze((pc - pc0) * output_r)
        loss_r = torch.mean(loss_1 + loss_2)
        loss = loss_r

        optimizer.zero_grad()
        loss.backward()
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
            pred = model(Z)
            pred = torch.squeeze(pred)
            pred = datau - pred
            UL2 = torch.norm(pred, p=2) / torch.norm(datau, p=2)
            print('epoch:', epoch, 'loss:', format(loss.item(), '.5f'), 'loss_r:','lr', format(optimizer.state_dict()['param_groups'][0]['lr'], '.8f'),
                  "UL2Rerror: ", format(UL2.item(), '.5f'))

        if epoch > int(3 * epochs / 5):
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                torch.save(model.state_dict(), 'model_S.mdl')

    print("Running time: ", time.time() - start)
    print('best epoch:', best_epoch, 'best loss:', best_loss)

if __name__ == '__main__':
    main()
