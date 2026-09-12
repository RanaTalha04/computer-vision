import torch
import torch.nn as nn
import torchvision.models as models

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
print(model.classifier)

for param in model.parameters():
    param.requires_grad = False

num_classes = 4
model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

print(f"ResNet18:        {count_params(resnet18):,} params")
print(f"EfficientNet-B0: {count_params(efficientnet_b0):,} params")
print(f"MobileNetV2:     {count_params(model):,} params")