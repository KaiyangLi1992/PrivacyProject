import torch
import torch.nn.functional as F
from torch import nn
import tqdm

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T

dataset = TUDataset(root='./tmp', name='NCI1', transform=T.NormalizeFeatures())
torch.manual_seed(12315)
dataset = dataset.shuffle()
dataset_length = len(dataset)

# split the dataset into 3 parts
DA_train = dataset[0:int(0.4 * dataset_length)]
D_aux = dataset[int(0.4 * dataset_length):int(0.7 * dataset_length)]
DA_test = dataset[int(0.7 * dataset_length):]

buck_num = 2

# load the model
model_path = "NCI_model.pt"

model_sage = torch.load(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_sage = model_sage.to(device)

# get graph embedding from node embedding for D aux using mean pool
D_aux_graph_embedding = []
D_aux_num_of_nodes_raw_data = []
D_aux_num_of_edges_raw_data = []
D_aux_graph_density_raw = []

for data in D_aux:
    data = data.to(device)
    test_output = model_sage(data.x, data.edge_index)
    D_aux_graph_embedding.append(test_output.sum(dim=0).cpu())
    D_aux_num_of_nodes_raw_data.append(data.num_nodes)
    D_aux_num_of_edges_raw_data.append(data.num_edges)
    density_temp = 2 * data.num_nodes / (data.num_edges * (data.num_edges - 1))
    D_aux_graph_density_raw.append(density_temp)

del test_output
del data

print(len(D_aux_num_of_nodes_raw_data))

D_aux_num_of_nodes_raw_data_train = D_aux_num_of_nodes_raw_data[0:int(0.7 * len(D_aux_num_of_nodes_raw_data))]
D_aux_num_of_edges_raw_data_train = D_aux_num_of_edges_raw_data[0:int(0.7 * len(D_aux_num_of_edges_raw_data))]
D_aux_graph_embedding_train = D_aux_graph_embedding[0:int(0.7 * len(D_aux_graph_embedding))]
D_aux_graph_density_train = D_aux_graph_density_raw[0:int(0.7 * len(D_aux_graph_density_raw))]
print(len(D_aux_num_of_nodes_raw_data_train))

D_aux_num_of_nodes_raw_data_test = D_aux_num_of_nodes_raw_data[int(0.7 * len(D_aux_num_of_nodes_raw_data)):]
D_aux_num_of_edges_raw_data_test = D_aux_num_of_edges_raw_data[int(0.7 * len(D_aux_num_of_edges_raw_data)):]
D_aux_graph_embedding_test = D_aux_graph_embedding[int(0.7 * len(D_aux_graph_embedding)):]
D_aux_graph_density_test = D_aux_graph_density_raw[int(0.7 * len(D_aux_graph_density_raw)):]
print(len(D_aux_num_of_nodes_raw_data_test))


def split_buck(data, buck):
    # get the maximum number in data
    N_max = data[0]

    for i in data:
        if i > N_max:
            N_max = i

    # sort the data w.r.t. data.num_nodes
    sorted_list = sorted(data, key=lambda x: x, reverse=False)
    # print('length of sorted list: ', len(sorted_list))

    # num of elements in each buck
    buck_num = round(len(data) / buck)

    # split point for data w.r.t. buck
    split_pt = []
    cnt = 0
    for i in sorted_list:
        cnt += 1
        if cnt == buck_num:
            cnt = 0
            split_pt.append(i)

    if len(split_pt) == buck:
        split_pt[-1] = N_max + 1
    else:
        split_pt.append(N_max + 1)

    res = []
    for i in data:
        for index, j in enumerate(split_pt):
            if i <= j:
                res.append(index)
                break

    return res


D_aux_nodes_train = split_buck(D_aux_num_of_nodes_raw_data_train, buck_num)
D_aux_nodes_test = split_buck(D_aux_num_of_nodes_raw_data_test, buck_num)

D_aux_edges_train = split_buck(D_aux_num_of_edges_raw_data_train, buck_num)
D_aux_edges_test = split_buck(D_aux_num_of_edges_raw_data_test, buck_num)

D_aux_density_train = split_buck(D_aux_graph_density_train, buck_num)
D_aux_density_test = split_buck(D_aux_graph_density_test, buck_num)


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


# mydataset = TrainSet(D_aux_graph_embedding_train, D_aux_nodes_train)
mydataset = TrainSet(D_aux_graph_embedding_train, D_aux_nodes_train, D_aux_edges_train, D_aux_density_train)
train_loader = DL(mydataset, batch_size=10, shuffle=True)


class Network(nn.Module):
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


model = Network(buck_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train():
    model.train()
    total_loss = 0

    for data in tqdm.tqdm(train_loader):
        inputs, num_nodes, num_edges, graph_density = data
        optimizer.zero_grad()

        inputs = inputs.detach()
        num_nodes = num_nodes.detach()
        num_edges = num_edges.detach()
        graph_density = graph_density.detach()

        pred_nodes, pred_edges, pred_density = model(inputs.to(device))
        loss_1 = criterion(pred_nodes, num_nodes.type(torch.LongTensor).to(device))
        loss_2 = criterion(pred_edges, num_edges.type(torch.LongTensor).to(device))
        loss_3 = criterion(pred_density, graph_density.type(torch.LongTensor).to(device))

        loss = loss_1 + loss_2 + loss_3

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(D_aux_nodes_train)


# mytestdata = TrainSet(D_aux_graph_embedding_test, D_aux_nodes_test)
mytestdata = TrainSet(D_aux_graph_embedding_test, D_aux_nodes_test, D_aux_edges_test, D_aux_density_test)
test_loader = DL(mytestdata, batch_size=1, shuffle=False)


@torch.no_grad()
def test():
    model.eval()
    total_loss = 0
    accu_nodes = 0
    accu_edges = 0
    accu_density = 0

    for data in tqdm.tqdm(test_loader):
        inputs, num_nodes, num_edges, graph_density = data
        inputs = inputs.detach()
        num_nodes = num_nodes.detach()
        num_edges = num_edges.detach()
        graph_density = graph_density.detach()

        pred_nodes, pred_edges, pred_density = model(inputs.to(device))

        pred_nodes_argmax = torch.argmax(pred_nodes, dim=1).type(torch.float).cpu()
        pred_edges_argmax = torch.argmax(pred_edges, dim=1).type(torch.float).cpu()
        pred_density_argmax = torch.argmax(pred_density, dim=1).type(torch.float).cpu()

        if torch.equal(pred_nodes_argmax, num_nodes):
            accu_nodes += 1

        if torch.equal(pred_edges_argmax, num_edges):
            accu_edges += 1

        if torch.equal(pred_density_argmax, graph_density):
            accu_density += 1

        loss_1 = criterion(pred_nodes, num_nodes.type(torch.LongTensor).to(device))

        loss_2 = criterion(pred_edges, num_edges.type(torch.LongTensor).to(device))

        loss_3 = criterion(pred_density, graph_density.type(torch.LongTensor).to(device))

        loss = loss_1 + loss_2 + loss_3

        total_loss += loss.item()

    return total_loss / len(D_aux_nodes_test), accu_nodes / len(D_aux_nodes_test), accu_edges / len(
        D_aux_nodes_test), accu_density / len(D_aux_nodes_test)


for epoch in range(60):
    train_loss = train()
    test_loss, test_nodes_accu, test_edges_accu, test_density_accu = test()
    print(
        f'Epoch: {epoch:4d}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Test node accu: {test_nodes_accu:.4f}, Test edge accu: {test_edges_accu:.4f}, Test density accu: {test_density_accu:.4f}')
