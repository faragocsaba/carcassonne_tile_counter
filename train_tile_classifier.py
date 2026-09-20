"""
===============================================================================
Carcassonne Tile Classifier - Advanced ResNet18 Training Script (train_tile_classifier.py)
===============================================================================
Dependencies:
    pip install torch torchvision pillow

Enhancements:
    - Transfer Learning via Pretrained ResNet18 (ImageNet weights)
    - Strong Data Augmentation (RandomAffine, ColorJitter, RandomErasing)
    - Solves Domain Shift / Edge Artifact Issues
===============================================================================
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# 1. Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10  # Pretrained ResNet18 converges extremely fast (10 epochs is plenty)
LEARNING_RATE = 0.0003  # Lower LR for fine-tuning pretrained weights
IMAGE_SIZE = (64, 64)
DATA_DIR = "generated_tiles"
MODEL_SAVE_PATH = "carcassonne_model.pth"
CLASSES_SAVE_PATH = "class_names.json"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Training] Using compute device: {device}")

# 2. Robust Data Augmentation Pipeline
# Simulates crop shifts, edge artifacts, lighting changes, and noise
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    # Slight random shift (+-5%) & scale (+-5%) to handle 1-2px crop offsets
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    # Color variation for lighting differences
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Randomly erase small rectangular patches to simulate selection borders/highlights
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), value='random')
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Dataset
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
num_classes = len(full_dataset.classes)
print(f"[Dataset] Found {len(full_dataset)} images across {num_classes} classes.")

# Save class index mapping
class_to_idx = full_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
with open(CLASSES_SAVE_PATH, "w") as f:
    json.dump(idx_to_class, f, indent=4)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Pretrained ResNet18 Architecture
def get_resnet_model(num_classes):
    # Load ImageNet pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace final Fully Connected layer for our 24 tile classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

model = get_resnet_model(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 5. Training Loop
print("\n[Training] Fine-tuning Pretrained ResNet18 Model...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = correct_train / total_train
    train_loss = running_loss / total_train

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = correct_val / total_val
    val_loss = val_loss / total_val

    print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# 6. Save Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n[Training] 🎉 Pretrained ResNet18 model saved to '{MODEL_SAVE_PATH}'.")
