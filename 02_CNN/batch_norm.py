import torch
import torch.nn as nn

x = torch.rand(8, 3, 32, 32)

conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
bn = nn.BatchNorm2d(num_features=16)

relu = nn.ReLU()

x = conv(x)
x = bn(x)
x = relu(x)

print(x.shape)