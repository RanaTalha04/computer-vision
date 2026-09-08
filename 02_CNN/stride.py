import torch
import torch.nn as nn

x = torch.randn(1, 3, 32, 32)

conv_stride1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
out1 = conv_stride1(x)

print(out1.shape)


conv_stride2 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
out2 = conv_stride2(x)

print(out2.shape)