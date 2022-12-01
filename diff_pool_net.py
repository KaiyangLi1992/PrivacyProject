from math import ceil

import torch
import torch.nn.functional as F

from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(0.25 * 111)
        self.gnn1_pool = DenseSAGEConv(37, num_nodes)
        self.gnn1_embed = DenseSAGEConv(37, 192)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = DenseSAGEConv(192, num_nodes)
        self.gnn2_embed = DenseSAGEConv(192, 192)

        self.gnn3_embed = DenseSAGEConv(192, 192)

        self.lin1 = torch.nn.Linear(192, 192)
        self.lin2 = torch.nn.Linear(192, 2)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        # print(s.shape)     # [1, 111, 28]
        # print(x.shape)     # [1, 111, 192]

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        # print(x.shape)     # [1, 28, 192]
        # print(adj.shape)   # [1, 28, 28]

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        # print(s.shape)
        # print(x.shape)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        node_embedding = x

        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2, node_embedding
