import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.randn(1, 3, 32, 32) # 1 batch of RGB image with size of 32x32 pixels 

conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
feature_maps = conv1(x)

print(feature_maps.shape)

# torch.Size([1, 16, 32, 32])
# 16 = number of feature maps (one per filter)
# 32x32 = spatial size of each feature map

single_map = feature_maps[0, 4].detach().numpy()  # batch 0, feature map index 4
plt.imshow(single_map, cmap='viridis')
plt.title("Feature map from filter #5")
plt.savefig("../images/feature_map.png")
print("Image saved successfully as feature_map.png")
