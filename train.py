import os
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ======================
# DEVICE
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# PATH (FIXED)
# ======================
dataset_path = r"D:\GradProject"

train_path = os.path.join(dataset_path, "Training")
test_path = os.path.join(dataset_path, "Testing")


print("TRAIN PATH:", train_path)
print("TEST PATH:", test_path)
print("TRAIN EXISTS:", os.path.exists(train_path))
print("TEST EXISTS:", os.path.exists(test_path))


# ======================
# LOAD DATA
# ======================
valid_ext = (".jpg", ".jpeg", ".png")

train_paths = glob(os.path.join(train_path, "**", "*.*"), recursive=True)
test_paths = glob(os.path.join(test_path, "**", "*.*"), recursive=True)

train_paths = [p for p in train_paths if p.lower().endswith(valid_ext)]
test_paths = [p for p in test_paths if p.lower().endswith(valid_ext)]

print("Train samples:", len(train_paths))
print("Test samples:", len(test_paths))


if len(train_paths) == 0:
    raise ValueError("No training images found! Check your folder structure.")


# ======================
# TRANSFORMS
# ======================
image_size = (224, 224)

train_transform = transforms.Compose([
    transforms.Resize(image_size),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=10,
        translate=(0.05,0.05),
        scale=(0.95,1.05)
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


# ======================
# DATASET
# ======================
class BrainDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

        self.labels = [os.path.basename(os.path.dirname(p)) for p in paths]
        self.classes = sorted(list(set(self.labels)))
        self.label2idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.label2idx[self.labels[idx]]

        if self.transform:
            img = self.transform(img)

        return img, label


# ======================
# SPLIT DATA
# ======================
dataset = BrainDataset(train_paths, transform=train_transform)

indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

test_dataset = BrainDataset(test_paths, transform=test_transform)


# ======================
# DATALOADERS
# ======================
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# ======================
# MODEL
# ======================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Linear(256, len(dataset.classes))
)

model = model.to(device)


# ======================
# LOSS / OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


# ======================
# TRAIN
# ======================
def train():
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss/len(train_loader), correct/total


# ======================
# VALIDATE
# ======================
def validate():
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss/len(val_loader), correct/total


# ======================
# TRAIN LOOP
# ======================
best = 0

for epoch in range(15):
    tl, ta = train()
    vl, va = validate()

    print(f"Epoch {epoch+1}")
    print("Train Acc:", ta)
    print("Val Acc:", va)

    if va > best:
        best = va
        torch.save(model.state_dict(), "best_model.pth")
        patience = 0

    else:
        patience += 1

        if patience == 5:
            print("Early Stopping")
            break


# ======================
# TEST
# ======================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)

        y_true.extend(y.numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))


# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()


import json

print(dataset.classes)

with open("classes.json","w") as f:
    json.dump(dataset.classes,f)