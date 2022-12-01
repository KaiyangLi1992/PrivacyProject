import torch.nn as nn


class MTLNetwork(nn.Module):
    def __init__(self, buck_num):
        super().__init__()
        self.num_features = 192
        self.num_buck = buck_num

        self.featureNet = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_features),
            nn.ReLU()
        )

        self.nodeNet = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, self.num_buck)
        )

        self.edgeNet = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, self.num_buck)
        )

        self.graphDensity = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, self.num_buck)
        )

    def forward(self, x):
        x = self.featureNet(x)
        pred_nodes = self.nodeNet(x)
        pred_edges = self.edgeNet(x)
        pred_density = self.graphDensity(x)

        return pred_nodes, pred_edges, pred_density
