import torch.nn as nn
import torchvision.models as models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False
    

num_classes = 4
model.fc = nn.Linear(in_features=512, out_features=num_classes)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params} / Total: {total_params}")
