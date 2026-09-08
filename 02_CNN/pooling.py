import torch
import torch.nn as nn

x = torch.randn(1, 16, 32, 32)

pool = nn.MaxPool2d(kernel_size=2, stride=2)
output = pool(x)

print(output.shape)  

avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
output_avg = avg_pool(x)

print(output_avg.shape)  
