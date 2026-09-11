import torch
import torchvision.models as models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

print(model)

x = torch.randn(1, 3, 224, 224) # Pre-trained models expect 224 x 224 input
output = model(x)
print(output.shape)