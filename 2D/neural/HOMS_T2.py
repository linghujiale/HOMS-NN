import torch
import torch.nn as nn
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

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

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def main():


    epsilon = 1/10


    # load N1 N2 NN model

    model_U0 = drrnn_t(3, 20, 3, 1).to(device)
    model_U0.load_state_dict(torch.load('model_t0.mdl'))
    model_U0.cpu()

    in_N = 2
    m = 30
    depth = 4
    out_N = 1

    model_N1 = drrnn(in_N, m, depth, out_N).to(device)
    model_N1.load_state_dict(torch.load('model_N1.mdl'))
    model_N1.cpu()

    model_N2 = drrnn(in_N, m, depth, out_N).to(device)
    model_N2.load_state_dict(torch.load('model_N2.mdl'))
    model_N2.cpu()

    model_N11 = drrnn(in_N, m, depth, out_N).to(device)
    model_N11.load_state_dict(torch.load('model_N11.mdl'))
    model_N11.cpu()

    model_N12 = drrnn(in_N, m, depth, out_N).to(device)
    model_N12.load_state_dict(torch.load('model_N12.mdl'))
    model_N12.cpu()

    model_N22 = drrnn(in_N, m, depth, out_N).to(device)
    model_N22.load_state_dict(torch.load('model_N22.mdl'))
    model_N22.cpu()

    model_N21 = drrnn(in_N, m, depth, out_N).to(device)
    model_N21.load_state_dict(torch.load('model_N21.mdl'))
    model_N21.cpu()

    model_S = drrnn(in_N, m, depth, out_N).to(device)
    model_S.load_state_dict(torch.load('model_S.mdl'))
    model_S.cpu()

    x1 = torch.linspace(0, 1, 500)
    x2 = torch.linspace(0, 1, 500)
    X, Y = torch.meshgrid(x1, x2)
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]

    XX = X / epsilon - (X / epsilon).floor()
    YY = Y / epsilon - (Y / epsilon).floor()
    pre_N1 = model_N1(torch.cat((XX, YY), dim=1))
    pre_N2 = model_N2(torch.cat((XX, YY), dim=1))
    pre_N11 = model_N11(torch.cat((XX, YY), dim=1))
    pre_N12 = model_N12(torch.cat((XX, YY), dim=1))
    pre_N22 = model_N22(torch.cat((XX, YY), dim=1))
    pre_N21 = model_N21(torch.cat((XX, YY), dim=1))
    pre_S = model_S(torch.cat((XX, YY), dim=1))

    csvt = 0.05
    csv = np.ones((250000, 1))
    for i in range(20):

        Z = X * 0 + csvt
        csvt = csvt + 0.05

        X.requires_grad_()
        Y.requires_grad_()
        Z.requires_grad_()
        pre_U0 = model_U0(torch.cat((X, Y, Z), dim=1))

        U0_x = autograd.grad(outputs=pre_U0, inputs=X,
                             grad_outputs=torch.ones_like(pre_U0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        U0_y = autograd.grad(outputs=pre_U0, inputs=Y,
                             grad_outputs=torch.ones_like(pre_U0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        U0_t = autograd.grad(outputs=pre_U0, inputs=Z,
                             grad_outputs=torch.ones_like(pre_U0),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        U0_xx = autograd.grad(outputs=U0_x, inputs=X,
                             grad_outputs=torch.ones_like(U0_x),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        U0_xy = autograd.grad(outputs=U0_x, inputs=Y,
                             grad_outputs=torch.ones_like(U0_x),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        U0_yy = autograd.grad(outputs=U0_y, inputs=Y,
                             grad_outputs=torch.ones_like(U0_y),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

        U1 = pre_U0 + epsilon * (pre_N1 * U0_x + pre_N2 * U0_y)

        U2 = U1 + (epsilon ** 2) * (pre_N11 * U0_xx + (pre_N12 + pre_N21) * U0_xy + pre_N22 * U0_yy + pre_S * U0_t)

        U2 = U2.detach().numpy()
        csv = np.hstack((csv, U2))

    # pre = np.hstack((N1, N2, N11, N12, N22, N21, U0, U1, U2))
    np.savetxt('T2_pre.csv',csv,delimiter=',')


if __name__ == '__main__':
    main()
