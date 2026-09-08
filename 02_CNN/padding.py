import torch
import torch.nn as nn

x = torch.randn(1, 3, 32, 32)

conv_no_pad = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
out_no_pad = conv_no_pad(x)

print(out_no_pad.shape)

conv_pad = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
out_pad = conv_pad(x)

print(out_pad.shape)