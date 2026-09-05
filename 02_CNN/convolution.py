import torch
import torch.nn as nn

conv = nn.Conv2d(

    in_channels=3, 
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1
)

x = torch.randint(1, 3, (1, 3, 32, 32)).float() # 1 - Low value, 3 - Max value (not included), (1 - Batch Size, 3 - Channels, 32 - Height, 32 - Width) - 4d image shape
output = conv(x)

print(output.shape)