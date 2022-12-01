import torch
from torch.utils.data import Dataset


class TrainSet(Dataset):
    def __init__(self, X, num_nodes, num_edges, graph_density):
        self.X = torch.stack(X)
        self.num_nodes = torch.tensor(num_nodes, dtype=torch.float)
        self.num_edges = torch.tensor(num_edges, dtype=torch.float)
        self.graph_density = torch.tensor(graph_density, dtype=torch.float)

    def __getitem__(self, index):
        return self.X[index], self.num_nodes[index], self.num_edges[index], self.graph_density[index]

    def __len__(self):
        return len(self.num_nodes)
