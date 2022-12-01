from math import ceil

import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import DenseSAGEConv, dense_mincut_pool


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=192, max_nodes=111):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        num_nodes = ceil(0.5 * max_nodes)
        self.pool1 = Linear(hidden_channels, num_nodes)

        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(hidden_channels, num_nodes)

        self.conv3 = DenseSAGEConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        # print(x.shape)  # 200 x 111 x 37
        # print(adj.shape)  # 200 x 111 x 111
        x = self.conv1(x, adj)
        s = self.pool1(x)
        # print(x.shape)    # 200 x 111 x 192
        # print(s.shape)    # 200 x 111 x 56

        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s)
        # print(x.shape)  # 200 x 56 x 192
        # print(adj.shape)    # 200 x 56 x 56

        x = self.conv2(x, adj)
        s = self.pool2(x)
        # print(x.shape)  # 200 x 56 x 192
        # print(s.shape)  # 200 x 56 x 28

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        # print(x.shape)  # 200 x 28 x 192
        # print(adj.shape)    # 200 x 28 x 28

        x = self.conv3(x, adj)
        # print(x.shape)  # 200 x 28 x 192

        # mean the last layer
        node_embedding = x.mean(dim=1)
        # print(x.shape)  # 200 x 192
        x = self.lin1(node_embedding).relu()
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), mc1 + mc2, o1 + o2, node_embedding
