import json
import urllib.request
import torchvision.models as models


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

weights = models.ResNet18_Weights.DEFAULT
categories = weights.meta['categories']

print(len(categories))
print(categories[:10])

import torch
x = torch.randn(1, 3, 224, 224)  # replace with a real preprocessed image
output = model(x)
predicted_idx = output.argmax(dim=1).item()
print("Predicted class:", categories[predicted_idx])