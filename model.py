import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
import math

node = 112
SE = 4
lamda = 0.5
smega = 1.2
d_const = 3


class EP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(EP, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (1, node), dtype=torch.float64)  # Ensure float64

    def forward(self, x):
        x = self.conv(x)
        return x


class NP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(NP, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (node, 1), dtype=torch.float64)  # Ensure float64

    def forward(self, x):
        x = self.conv(x)
        return x

class GPC_diff(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GPC_diff, self).__init__()
        self.out_dim = out_dim
        # Adjust kernel size to (1, 3) to match input dimensions
        self.conv = nn.Conv2d(in_dim, out_dim, (1, d_const), dtype=torch.float64)

    def forward(self, x):
        batchsize = x.shape[0]
        x_c = self.conv(x)
        x_C = x_c.expand(batchsize, self.out_dim, node, d_const)  # Adjusted to expand to the correct size
        x_R = x_C.permute(0, 1, 3, 2)  # Permute to [batchsize, out_dim, 3, 112]

        # Permute back to match x_C's shape
        x_R = x_R.permute(0, 1, 3, 2)
        x = x_C + x_R  # Ensure dimensions match for addition
        return x


class EP_diff(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(EP_diff, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (1, d_const), dtype=torch.float64)  # Ensure float64

    def forward(self, x):
        x = self.conv(x)
        return x


class BC_GCN_diff(nn.Module):
    def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim):
        super(BC_GCN_diff, self).__init__()

        # Adjust input channels based on the number of diffusion components or other input channels
        self.GPC_1 = GPC_diff(1, GPC_dim_1)
        self.GPC_2 = GPC_diff(GPC_dim_1, GPC_dim_2)
        self.GPC_3 = GPC_diff(GPC_dim_2, GPC_dim_3)

        self.EP = EP_diff(GPC_dim_3, EP_dim)
        self.NP = NP(EP_dim, NP_dim)

        # Update flattened size calculation based on NP output
        flattened_size = NP_dim  # Update if the size of the output changes

        self.fc = nn.Linear(flattened_size, 1, dtype=torch.float64)  # Ensure float64

    def forward(self, x):
        x = self.GPC_1(x)
        x = F.relu(x)

        x = self.GPC_2(x)
        x = F.relu(x)

        x = self.GPC_3(x)
        x = F.relu(x)

        x = self.EP(x)
        x = F.relu(x)

        x = self.NP(x)
        x = F.relu(x)

        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
