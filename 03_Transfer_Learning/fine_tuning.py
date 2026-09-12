import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Assumes folder structure: data/train/class1, data/train/class2, ...

train_dataset = ImageFolder("data/train", transform=train_transform)
val_dataset   = ImageFolder("data/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes found:", train_dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 2. Load pretrained model + freeze backbone (Concept 1 & 2)
# ---------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

# Replace the final classification layer for OUR classes
model.fc = nn.Linear(in_features=512, out_features=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()


# ---------------------------------------------------------
# 3. Reusable train/eval loop (same theory as Phase 2)
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(loader), 100 * correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# ---------------------------------------------------------
# 4. STAGE 1: Feature extraction - train only the new fc layer
# ---------------------------------------------------------
print("\n--- Stage 1: Feature Extraction (backbone frozen) ---")
optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_stage1)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f} train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%")


# ---------------------------------------------------------
# 5. STAGE 2: Fine-tuning - unfreeze last block, small LR
# ---------------------------------------------------------
print("\n--- Stage 2: Fine-tuning (layer4 unfrozen) ---")
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer_stage2 = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},  # small LR - already-good weights
    {'params': model.fc.parameters(), 'lr': 0.001}          # normal LR - still learning
])

for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_stage2)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f} train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%")