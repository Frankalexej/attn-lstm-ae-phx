import torch
from torch.autograd import Function, Variable
import torch.nn.functional as Func
from torch.nn import Module, Parameter
import torch.nn as nn
import math

class DoubleLin(Module): 
    def __init__(self, n_in, n_mid, n_out): 
        super(DoubleLin, self).__init__()
        self.lin1 = nn.Linear(n_in, n_mid)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(n_mid, n_out)
        # self.batch_norm = nn.BatchNorm1d(num_features=n_out)
    
    def forward(self, x): 
        x = self.lin2(self.relu(self.lin1(x)))
        return x
    
# class LinearPack(nn.Module):
#     def __init__(self, in_dim, out_dim, dropout_rate=0.5):
#         super(LinearPack, self).__init__()

#         self.linear = nn.Linear(in_dim, out_dim)
#         self.relu = nn.ReLU()
#         # self.relu = nn.LeakyReLU()
#         # self.relu = nn.Tanh()
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.linear(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         return x

class LinearPack(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.5):
        super(LinearPack, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # self.norm = nn.LayerNorm(out_dim)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        # self.relu = nn.Tanh()
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        # x = self.layernorm(x)  # Apply LayerNorm after linear transformation
        # x = self.norm(x)
        x = self.relu(x)
        # x = self.dropout(x)  # Apply dropout after activation (if using dropout)
        return x