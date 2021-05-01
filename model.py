import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MODEL_SIGMOID(nn.Module):
    def __init__(self,hidden):
        super(MODEL_SIGMOID,self).__init__()

        self.block = nn.Sequential(
            nn.Linear(2,hidden),
            nn.Sigmoid(),
            nn.Linear(hidden,1)
        )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                print("SISSISISISISISISISISISISISI")
                torch.nn.init.normal_(m.weight.data)

    def forward(self,x):
        return self.block(x)

class MODEL_TANH(nn.Module):
    def __init__(self,hidden):
        super(MODEL_TANH,self).__init__()

        self.block = nn.Sequential(
            nn.Linear(2,hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden,1)
        )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                print("SISSISISISISISISISISISISISI")
                torch.nn.init.normal_(m.weight.data)

    def forward(self,x):
        return self.block(x)

