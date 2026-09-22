import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Load a pretrained Faster R-CNN (notice: uses a ResNet50 backbone -
# directly connects back to Phase 3's transfer learning concepts!)
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

img = Image.open("bus.jpg").convert("RGB")
img_tensor = to_tensor(img)  # Faster R-CNN expects a plain tensor, not a batch necessarily

with torch.no_grad():
    predictions = model([img_tensor])  # note: takes a LIST of images

print(predictions[0].keys())  # dict_keys(['boxes', 'labels', 'scores'])
print("Number of boxes found:", len(predictions[0]['boxes']))