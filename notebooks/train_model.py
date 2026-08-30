# ============================================================
# Handwritten-to-Digital: Model Training Script
# Run this in Google Colab (Runtime > Change runtime type > GPU)
# ============================================================
# This trains a single CNN to classify:
#   - Digits 0-9          (from EMNIST 'balanced' split)
#   - Letters A-Z, a-z... (from EMNIST 'balanced' split, merges some
#                           visually-similar upper/lower pairs -> 47 classes total)
#   - Math symbols        (+, -, *, /, =, ., (, ) etc. from Kaggle dataset)
#
# Final combined class count: 47 (EMNIST balanced) + N (math symbols)
# ============================================================

# ---- STEP 0: Install/imports ----
# !pip install torch torchvision opencv-python-headless kaggle -q

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from torchvision import transforms
import numpy as np
import os
import cv2
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# STEP 1: Load EMNIST (digits + letters), split='balanced'
# ============================================================
# EMNIST balanced has 47 classes:
# 0-9 (digits), then A-Z, then a subset of lowercase letters
# (some lowercase merged with uppercase since they look identical
#  in handwriting, e.g. 'c', 'o', 's', 'v', 'w', 'x', 'z')

# IMPORTANT: EMNIST images are rotated/flipped by default due to how
# the dataset was stored -- we fix that in the transform below.

emnist_transform = transforms.Compose([
    lambda img: transforms.functional.rotate(img, -90),
    lambda img: transforms.functional.hflip(img),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

emnist_train = torchvision.datasets.EMNIST(
    root="./data", split="balanced", train=True,
    download=True, transform=emnist_transform
)
emnist_test = torchvision.datasets.EMNIST(
    root="./data", split="balanced", train=False,
    download=True, transform=emnist_transform
)

# EMNIST balanced label mapping (47 classes) -> human-readable characters
EMNIST_BALANCED_MAPPING = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
# NOTE: verify this mapping against emnist-balanced-mapping.txt included
# in the downloaded EMNIST files -- torchvision stores it at
# ./data/EMNIST/raw/emnist-balanced-mapping.txt
# Each line is: <label_index> <ASCII code>
# We build the mapping programmatically below to avoid hardcoding errors:

mapping_path = "./data/EMNIST/raw/emnist-balanced-mapping.txt"
emnist_classes = {}
if os.path.exists(mapping_path):
    with open(mapping_path) as f:
        for line in f:
            idx, ascii_code = line.strip().split()
            emnist_classes[int(idx)] = chr(int(ascii_code))
else:
    print("WARNING: mapping file not found, using fallback list (verify manually)")
    emnist_classes = {i: c for i, c in enumerate(EMNIST_BALANCED_MAPPING)}

NUM_EMNIST_CLASSES = len(emnist_classes)
print(f"EMNIST balanced classes: {NUM_EMNIST_CLASSES}")
print(emnist_classes)

# ============================================================
# STEP 2: Load Math Symbols dataset (from Kaggle)
# ============================================================
# Dataset: https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
# Download via Kaggle API (requires kaggle.json uploaded to Colab):
#
#   from google.colab import files
#   files.upload()  # upload kaggle.json here
#   !mkdir -p ~/.kaggle
#   !cp kaggle.json ~/.kaggle/
#   !chmod 600 ~/.kaggle/kaggle.json
#   !kaggle datasets download -d xainano/handwrittenmathsymbols
#   !unzip -q handwrittenmathsymbols.zip -d math_symbols
#
# The dataset is organized as folders per symbol, e.g.:
#   math_symbols/+/*.jpg
#   math_symbols/-/*.jpg
#   math_symbols/times/*.jpg
#   math_symbols/=/*.jpg
#   ...
#
# We only keep a curated subset of symbols relevant to basic expressions
# (avoids exploding class count with rare symbols like integrals, sums, etc.)

RELEVANT_MATH_SYMBOLS = {
    "+": "+", "-": "-", "times": "*", "div": "/", "=": "=",
    "(": "(", ")": ")", "decimal": ".",
}

class MathSymbolsDataset(Dataset):
    """Loads images from folder-per-class structure, filtered to RELEVANT_MATH_SYMBOLS."""
    def __init__(self, root_dir, class_offset, transform=None):
        self.samples = []
        self.transform = transform
        self.class_offset = class_offset
        self.label_to_symbol = {}

        next_label = 0
        for folder_name, symbol in RELEVANT_MATH_SYMBOLS.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"  (skip) folder not found: {folder_path}")
                continue
            label = class_offset + next_label
            self.label_to_symbol[label] = symbol
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(folder_path, fname), label))
            next_label += 1

        self.num_classes = next_label
        print(f"Loaded {len(self.samples)} math symbol images across {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        img = 255 - img  # invert if needed so strokes are white on black (match EMNIST convention)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # add channel dim
        if self.transform:
            img = self.transform(img)
        return img, label

MATH_SYMBOLS_ROOT = "./math_symbols"  # adjust after unzip
math_dataset = MathSymbolsDataset(MATH_SYMBOLS_ROOT, class_offset=NUM_EMNIST_CLASSES)

NUM_MATH_CLASSES = math_dataset.num_classes
TOTAL_CLASSES = NUM_EMNIST_CLASSES + NUM_MATH_CLASSES
print(f"Total combined classes: {TOTAL_CLASSES}")

# Build the final class_map: label -> character (for use at inference time)
class_map = {}
class_map.update(emnist_classes)
class_map.update(math_dataset.label_to_symbol)

with open("class_map.json", "w") as f:
    json.dump(class_map, f)
print("Saved class_map.json")

# Split math_dataset into train/test (90/10)
math_train_size = int(0.9 * len(math_dataset))
math_test_size = len(math_dataset) - math_train_size
math_train, math_test = torch.utils.data.random_split(
    math_dataset, [math_train_size, math_test_size],
    generator=torch.Generator().manual_seed(42)
)

# ============================================================
# STEP 3: Combine datasets
# ============================================================
train_dataset = ConcatDataset([emnist_train, math_train])
test_dataset = ConcatDataset([emnist_test, math_test])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

# ============================================================
# STEP 4: Define CNN model
# ============================================================
class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        # After 3 pools on 32x32 input: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CharCNN(TOTAL_CLASSES).to(device)
print(model)

# ============================================================
# STEP 5: Train
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

EPOCHS = 15

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    scheduler.step()

    train_loss = running_loss / len(train_dataset)
    test_acc = evaluate(test_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

# ============================================================
# STEP 6: Save model + class map
# ============================================================
torch.save(model.state_dict(), "char_cnn.pt")
print("Saved char_cnn.pt")
print("Download char_cnn.pt and class_map.json from Colab's file browser,")
print("then place them in backend/model/ for the Flask app.")
