import torch
import torch.nn as nn
from torch import optim, autograd
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
        data_x, data_y = torch.chunk(x, 2, 1)
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x * data_x * data_y * (1 - data_x) * (1 - data_y)


def get_interior_points_a(N,d):
    """
    N is the number of point, d is the dimension of point
    """
    return torch.rand(N,d)

def get_point(N):
    x1 = torch.linspace(0, 1, N)
    x2 = torch.linspace(0, 1, N)
    X, Y = torch.meshgrid(x1, x2)
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    return Z



def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def func_thermal(X, conductivity1, conductivity2, func):
    # conductivity1 is out thermal conductivity conductivity2 is inner
    k = 10000  # k 越大，间断越小
    r = 0.3  # 圆半径
    x, y = torch.chunk(X, 2, 1)
    if func == "conductivity":
        output = (1 / (1 + torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 - r ** 2)))) * (conductivity1 - conductivity2) + conductivity2
    if func == "partial_x":
        output = (conductivity1 - conductivity2) * (
                    (2 * k * (x - 0.5) * torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 - r ** 2))) / (
                        (1 + torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 - r ** 2))) ** 2))
    if func == "partial_y":
        output = (conductivity1 - conductivity2) * (
                    (2 * k * (y - 0.5) * torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 - r ** 2))) / (
                        (1 + torch.exp(- k * ((x - 0.5) ** 2 + (y - 0.5) ** 2 - r ** 2))) ** 2))
    return output

def quasirandom(n_samples, sampler):

    distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
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
    E1 = 30000
    E2 = 600
    v1 = 0.25
    v2 = 0.25
    pc1 = 50
    pc2 = 5
    T0 = 37.315
    epochs = 10000

    in_N = 2
    m = 30
    depth = 4
    out_N = 1

    data = pd.read_csv(r"data\N1.csv")
    data = np.array(data)
    data = data[0,0:10000]
    data = data.reshape(100, 100)
    datau = torch.tensor(data).to(device)
    # xr = get_point(150)

    xr = quasirandom(30000, "sobol")
    thermalxr = func_thermal(xr, thermal1, thermal2, "conductivity")
    xr = xr.to(device)
    thermalxr = thermalxr.to(device)
    xr.requires_grad_()
    best_loss, best_epoch = 100010, 0
    stat = time.time()

    model = drrnn(in_N, m, depth, out_N).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.8)
    print(model)

    for epoch in range(epochs+1):

        output_r = model(xr)
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_x, u_y = torch.chunk(grads, 2, 1)
        loss_r1 = 0.5 * torch.sum(thermalxr * torch.pow(grads, 2), dim=1)
        loss_r2 = torch.squeeze(thermalxr * u_x)
        loss_r = torch.mean(loss_r1 + loss_r2)
        # loss_r = torch.mean(loss_r)
        loss = loss_r

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            x1 = torch.linspace(0, 1, 100)
            x2 = torch.linspace(0, 1, 100)
            X, Y = torch.meshgrid(x1, x2)
            Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
            Z = Z.to(device)
            pred = model(Z)
            pred = pred.reshape(100, 100)
            error_u = datau - pred
            UL2 = torch.norm(error_u, p=2) / torch.norm(datau, p=2)

            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:', loss_r.item(),
                  'lr', optimizer.state_dict()['param_groups'][0]['lr'],
                  "NL2Rerror: ", UL2.item())

    torch.save(model.state_dict(), 'model_N1.mdl')
    print("Running time: ", time.time()-stat)

if __name__ == '__main__':
    main()
