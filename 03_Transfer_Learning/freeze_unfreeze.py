import torch
import torch.nn as nn
import torchvision.models as models


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Renet Structure
# conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc

# 1. Freezing params
for param in model.parameters():
    param.requires_grad = False
    
    
# Unfreeze params
for param in model.layer4.parameters():
    param.requires_grad = True
    
    
for name, param in model.named_parameters():
    print(name, "-> trainables: ", param.requires_grad)
    
    
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")