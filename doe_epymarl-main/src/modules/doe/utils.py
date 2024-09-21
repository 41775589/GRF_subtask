import torch
from torch.utils.data import Dataset
from itertools import chain
import torch.nn as nn
from typing import Iterable


class SimpleListDataset(Dataset):
        
    def __init__(self, x, y):
        # x
        if isinstance(x, torch.Tensor):
            self.x_train = torch.clone(x).detach().float()
        else:
            self.x_train = torch.tensor(x, dtype=torch.float32)
        # y
        if isinstance(y, torch.Tensor):
            self.y_train = torch.clone(y).detach().float()
        else:
            self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]



def fc_network(layer_dims: Iterable[int], init_ortho=True):
    """
    Builds a fully-connected NN with ReLU activation functions.

    """
    if init_ortho: 
        init = init_orthogonal
    else:
        init = lambda m: m

    network = nn.Sequential(
                *chain(
                    *((init(nn.Linear(layer_dims[i], layer_dims[i+1])),
                       nn.ReLU())
                      for i in range(len(layer_dims)-1))
                    ),
                )
    del network[-1]  # remove the final ReLU layer
    return network


def init_orthogonal(m):
    nn.init.orthogonal_(m.weight)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)
    return m
