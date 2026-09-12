import torch.nn as nn
import torchvision.models as models

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
print(model.classifier)

for param in model.parameters():
    param.requires_grad = False
    

num_classes = 4
model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)


def count_param(m):
    return sum(p.numel() for p in m.parameters())

print(f"EfficientNet-B0: {count_param(model):,} parameters")