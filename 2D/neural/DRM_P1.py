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
sys.stdout = Logger('uvb1.txt')

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

def get_interior_points_c(N):

    x = torch.linspace(0, 1, N)
    X, Y = torch.meshgrid(x, x)
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


def Cmatrix(E, V):

    CC = np.array([[E/(1-V**2), (E * V ) / (1 - V ** 2), 0],
                  [(E * V) / (1 - V ** 2), E / (1 - V ** 2), 0],
                  [0, 0, E / (2 * (1 + V))]])
    return CC

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
    E1 = 300
    E2 = 6
    v1 = 0.25
    v2 = 0.25
    pc1 = 50
    pc2 = 5
    T0 = 37.315
    epochs = 10000

    in_N = 2
    m = 30
    depth = 4
    out_N = 2

    data = pd.read_csv(r"data\ub1.csv")
    data = np.array(data)
    data = data[0,0:10000]
    data = data.reshape(100, 100)
    datau = torch.tensor(data).to(device)
    print("max data: ", format(torch.max(datau), '.5f'), " max data: ", format(torch.min(datau), '.5f'))

    data = pd.read_csv(r"data\vb1.csv")
    data = np.array(data)
    data = data[0,0:10000]
    data = data.reshape(100, 100)
    datav = torch.tensor(data).to(device)
    print("max data: ", format(torch.max(datav), '.5f'), " max data: ", format(torch.min(datav), '.5f'))

    C1 = Cmatrix(E1, v1)
    C2 = Cmatrix(E2, v2)
    print("C1: ", C1)
    print("C2: ", C2)

    # load lower-order UC
    model_N1 = drrnn(in_N, m, depth, 1).to(device)
    model_N1.load_state_dict(torch.load('model_N1.mdl'))

    model_uvb = drrnn(in_N, m, depth, out_N).to(device)
    model_uvb.load_state_dict(torch.load('model_P0.mdl'))

    # calculate C0
    xr = get_interior_points_c(300)
    xr = xr.to(device)
    xr.requires_grad_()
    beta = func_thermal(xr, beta1, beta2, "conductivity")
    C1111 = func_thermal(xr, C1[0, 0], C2[0, 0], "conductivity")
    C1122 = func_thermal(xr, C1[0, 1], C2[0, 1], "conductivity")
    C1212 = func_thermal(xr, C1[2, 2], C2[2, 2], "conductivity")

    output = model_uvb(xr)
    output_u, output_v = torch.chunk(output, 2, 1)
    grads_u = autograd.grad(outputs=output_u, inputs=xr,
                            grad_outputs=torch.ones_like(output_u),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    u_x, u_y = torch.chunk(grads_u, 2, 1)
    grads_v = autograd.grad(outputs=output_v, inputs=xr,
                            grad_outputs=torch.ones_like(output_v),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    v_x, v_y = torch.chunk(grads_v, 2, 1)
    beta11 = torch.mean(beta + C1111 * u_x + C1122 * v_y)
    print("beta11: ", beta11)

    # create higher-order UC
    model = drrnn(in_N, m, depth, out_N).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.8)
    print(model)

    # generate the data set
    xr = quasirandom(40000, "sobol")
    xr = xr.to(device)
    C1111 = func_thermal(xr, C1[0, 0], C2[0, 0], "conductivity")
    C1122 = func_thermal(xr, C1[0, 1], C2[0, 1], "conductivity")
    C1212 = func_thermal(xr, C1[2, 2], C2[2, 2], "conductivity")
    beta = func_thermal(xr, beta1, beta2, "conductivity")
    xr.requires_grad_()
    xrx, xry = torch.chunk(xr, 2, 1)

    best_loss, best_epoch = 100010, 0
    stat = time.time()
    for epoch in range(epochs+1):

        uv = model(torch.cat([xrx, xry], dim=1))
        u, v = torch.chunk(uv, 2, 1)
        grads_u = autograd.grad(outputs=u, inputs=xr,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        u_x, u_y = torch.chunk(grads_u, 2, 1)
        grads_v = autograd.grad(outputs=v, inputs=xr,
                              grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        v_x, v_y = torch.chunk(grads_v, 2, 1)

        luv = model_uvb(torch.cat([xrx, xry], dim=1))
        lu, lv = torch.chunk(luv, 2, 1)
        grads_u = autograd.grad(outputs=lu, inputs=xr,
                              grad_outputs=torch.ones_like(lu),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        lu_x, lu_y = torch.chunk(grads_u, 2, 1)
        grads_v = autograd.grad(outputs=lv, inputs=xr,
                              grad_outputs=torch.ones_like(lv),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        lv_x, lv_y = torch.chunk(grads_v, 2, 1)
        ln = model_N1(xr)

        loss_r1 = 0.5 * (torch.squeeze(C1111 * u_x * u_x) + torch.squeeze(C1122 * v_y * u_x) + torch.squeeze(C1122 * u_x * v_y)
                         + torch.squeeze(C1111 * v_y * v_y) + 2 * torch.squeeze(C1212 * u_y * v_x) + torch.squeeze(C1212 * u_y * u_y)
                         + torch.squeeze(C1212 * v_x * v_x))
        loss_r2 = torch.squeeze((beta - beta11) * u)
        loss_r3 = torch.squeeze((C1111 * lu_x + C1122 * lv_y) * u)
        loss_r4 = torch.squeeze((lu_y + lv_x) * C1212 * v)
        loss_r5 = torch.squeeze((C1111 * u_x + C1122 * v_y) * lu)
        loss_r6 = torch.squeeze((u_y + v_x) * C1212 * lv)
        loss_r7 = torch.squeeze((u_x + v_y) * beta * ln)
        loss_r = torch.mean(loss_r1 - loss_r2 - loss_r3 - loss_r4 + loss_r5 + loss_r6 + loss_r7)
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
            pred_u, pred_v = torch.chunk(pred, 2, 1)
            pred_u = pred_u.reshape(100, 100) / 100
            error_u = datau - pred_u
            UL2 = torch.norm(error_u, p=2) / torch.norm(datau, p=2)
            pred_v = pred_v.reshape(100, 100) / 100
            error_v = datav - pred_v
            VL2 = torch.norm(error_v, p=2) / torch.norm(datav, p=2)
            print('epoch:', epoch, 'loss:', format(loss.item(), '.5f'), 'loss_r:', 'lr',
                  format(optimizer.state_dict()['param_groups'][0]['lr'], '.8f'),
                  "UL2Rerror: ", format(UL2.item(), '.5f'), "VL2Rerror: ", format(VL2.item(), '.5f'))

    torch.save(model.state_dict(), 'model_P1.mdl')
    print("Running time: ", time.time()-stat)

if __name__ == '__main__':
    main()
