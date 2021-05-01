import numpy as np

from plot_data import plot_output
from model import MODEL_SIGMOID
from torch.utils.data import DataLoader,TensorDataset

model = MODEL_SIGMOID(hidden=56)
import torch
model = torch.load("./data/FOLD1/CV2/model")
print(model)


if torch.cuda.is_available():
    device = torch.device("cuda:0");
    print("using cuda:0")
else:
    device = torch.device("cpu");
    print("using CPU")
print("Device ? : {0}".format(device))
model.to(device)
trainset = np.loadtxt("./data/FOLD1/CV1/valset.txt")
train_xy= torch.from_numpy(trainset[:, 0:2]).float()
train_z = torch.from_numpy(trainset[:, 2]).float().view(-1, 1)
trainset = TensorDataset(train_xy,train_z)
trainloader = DataLoader(trainset, shuffle=True)
model.eval()
for idx, data in enumerate(trainloader):
    input, label = data[0].to(device), data[1].to(device)
    output = model(input)
    dd = input.cpu().data.numpy()
    d = output.cpu().data.numpy()
    c = np.hstack([dd, d])
    if idx != 0:
        result_ = np.vstack([c, result_])
    else:
        result_ = c
plot_output(result_, "Hidden : ","./tt.png")