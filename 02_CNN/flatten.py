import torch
import torch.nn as nn

x = torch.randn(1, 3, 16, 16)

flatten = nn.Flatten()
output = flatten(x)

print(output.shape)