import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch.utils.data import DataLoader, Dataset

from diff_pool_net import Net

max_nodes = 111


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


dataset = TUDataset(root='/tmp/NCI1', name='NCI1', transform=T.ToDense(max_nodes))
torch.manual_seed(12315)
dataset = dataset.shuffle()
dataset_length = len(dataset)

# split dataset into 3 parts
DA_train = dataset[0:int(0.4 * dataset_length)]
D_aux = dataset[int(0.4 * dataset_length):int(0.7 * dataset_length)]
DA_test = dataset[int(0.7 * dataset_length):]

DA_train_train = DA_train[0:int(0.7 * len(DA_train))]
DA_train_val = DA_train[int(0.7 * len(DA_train)):]


class TrainSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index].x, self.data[index].adj, self.data[index].y

    def __len__(self):
        return len(self.data)


train_dataset = TrainSet(DA_train_train)
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)

test_dataset = TrainSet(DA_train_val)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    loss_all = 0

    for data in train_loader:
        # data = data.to(device)
        x = data[0].squeeze().to(device)
        adj = data[1].squeeze().to(device)
        y = data[2].squeeze(dim=0).to(device)

        optimizer.zero_grad()
        output, _, _, _ = model(x, adj)
        loss = F.nll_loss(output, y.view(-1))
        loss.backward()
        loss_all += float(loss)
        optimizer.step()
    return loss_all / len(DA_train_train)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        # data = data.to(device)
        x = data[0].squeeze().to(device)
        adj = data[1].squeeze().to(device)
        y = data[2].squeeze(dim=0).to(device)

        pred = model(x, adj)[0].max(dim=1)[1]
        correct += int(pred.eq(y.view(-1)).sum())
    return correct / len(DA_train_val)


best_val_acc = test_acc = 0
for epoch in range(1, 500):
    train_loss = train()
    # print(epoch, train_loss)

    val_acc = test(test_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

data_save_path = "NCI_model_diff_pool.pt"
torch.save(model, data_save_path)
