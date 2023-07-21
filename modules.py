import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

class GIN(nn.Module):
    def __init__(self, args, task="node"):
        super().__init__()
        n_feat, n_hidden, n_layers = args.dim_in, args.dim_hid, args.n_layers
        n_class = args.dim_out
        self.conv1 = GINConv(
            Sequential(Linear(n_feat, n_hidden), BatchNorm1d(n_hidden), ReLU(),
                       Linear(n_hidden, n_hidden), ReLU()))

        self.convs = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(Linear(n_hidden, n_hidden),
                               BatchNorm1d(n_hidden),
                               ReLU(),
                               Linear(n_hidden, n_hidden),
                               ReLU()))
            )
        self.pool = False if task == "node" else True

        self.lin1 = Linear(n_hidden, n_hidden)
        self.lin2 = Linear(n_hidden, n_class)

        self.sigmoid = True

    def forward(self, x, edge_index, batch=None, embedding=False):
        x = x.float()
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        if self.pool:
            x = global_add_pool(x, batch)

        x = self.lin1(x).relu()
        if embedding:
            return x
        x = self.lin2(x)

        if not self.sigmoid:
            return x
        return x.sigmoid()

