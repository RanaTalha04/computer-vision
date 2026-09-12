import torch
import torchvision.models as models


resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

def count_params(model):
    return sum(p.numel() for p in model.parameters())


print(f"ResNet18: {count_params(resnet18):,} parameters")
print(f"ResNet50: {count_params(resnet50):,} parameters")

print(resnet18.layer1[0])
