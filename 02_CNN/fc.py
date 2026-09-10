## Fully-connected Layer.

import torch
import torch.nn as nn

x = torch.randn(1, 1024)

fc1 = nn.Linear(in_features=1024, out_features=128)
fc2 = nn.Linear(in_features=128, out_features=10)

x = fc1(x)
x = torch.relu(x)
x = fc2(x)

print(x.shape)