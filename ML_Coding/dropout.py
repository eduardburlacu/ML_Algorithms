import torch
from torch import nn
dropout = nn.Dropout(p=0.5)
x = torch.randn(3, 2, 7)
y = dropout(x)
print(y)