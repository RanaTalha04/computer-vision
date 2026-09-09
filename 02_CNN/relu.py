import torch
import torch.nn as nn

x = torch.tensor([[-2.0, 3.0, -0.5, 7.0]])

relu = nn.ReLU()
output = relu(x)

print(output)

x_img = torch.rand(1, 3, 32, 32)
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

feature_map = conv(x_img)
activated = relu(feature_map)
print(activated.shape)