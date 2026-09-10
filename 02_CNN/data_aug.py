import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip left-right
    transforms.RandomRotation(degrees=10), # rotate randomly up to 10 degrees 
    transforms.RandomCrop(32, padding=4),  # random crop with padding
    transforms.ToTensor(), # convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],    # normalize 
                          std=[0.5, 0.5, 0.5])
])

# IMPORTANT: the test/validation set should NOT be augmented -
# only normalized and converted, so you evaluate on real, unmodified images

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])