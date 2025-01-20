import torch
import torch.nn as nn
from torch import optim, autograd
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
sys.stdout = Logger('N13.txt')

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
    """
    randomly sample N points from interior of [0,1]^d
    """
    return torch.rand(N, d)

def get_interior_points_c(N):
    x = torch.linspace(0, 1, N)
    X, Y, Z = torch.meshgrid(x, x, x)
    Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]), dim=1)

    return Z

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
    pc1 = 45
    pc2 = 4.5
    epochs =10000

    in_N = 3
    m = 30
    depth = 5
    out_N = 1

    data = pd.read_csv(r"E:\pycharm\project6\3d\data\N13.csv")
    data = np.array(data)
    data = data[0, 0:27000]
    datau = torch.tensor(data).to(device)
    print("max data: ", torch.max(datau))
    datauL2 = torch.norm(datau, p=2)

    model_low = drrnn(in_N, m, out_N, depth).to(device)
    model_low.load_state_dict(torch.load('model_N3.mdl'))

    model = drrnn(in_N, m, out_N, depth).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.9)
    print(model)

    xr = quasirandom(200000, "halton")
    # xr = get_interior_points_a(30000, 3)
    xr = xr.to(device)
    thermal = func_thermal(xr, thermal1, thermal2, "conductivity")
    xr.requires_grad_()
    xrx, xry, xrz = torch.chunk(xr, 3, 1)

    # generate the data set
    best_loss, best_epoch = 10000, 0
    start = time.time()
    for epoch in range(epochs + 1):

        output_low = model_low(torch.cat([xrx, xry, xrz], dim=1))
        low_x = autograd.grad(outputs=output_low, inputs=xrx,
                              grad_outputs=torch.ones_like(output_low),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        output_r = model(torch.cat([xrx, xry, xrz], dim=1))
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_x, u_y, u_z = torch.chunk(grads, 3, 1)

        loss_1 = 0.5 * torch.sum(thermal * torch.pow(grads, 2), dim=1)
        loss_2 = torch.squeeze((low_x * output_r - output_low * u_x) * thermal)
        loss_r = torch.mean(loss_1 - loss_2)
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
            pred = model(Z)
            pred = torch.squeeze(pred)
            pred = datau - pred
            NL2 = torch.norm(pred, p=2) / datauL2

            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:', loss_r.item(),
                  'lr', optimizer.state_dict()['param_groups'][0]['lr'], "NL2Rerror: ", NL2.item(), "min: ", torch.min(output_r).item())

        if epoch > int(3 * epochs / 5):
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                torch.save(model.state_dict(), 'model_N13.mdl')

    print("Running time: ", time.time() - start)
    print('best epoch:', best_epoch, 'best loss:', best_loss)

if __name__ == '__main__':
    main()
