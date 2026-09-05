import torch
import torch.nn as nn

conv = nn.Conv2d(
    in_channels=3,
    out_channels=4, 
    kernel_size=3
)

print(conv.weight.shape)
print(conv.weight[0])
