import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset import OCRDataset
from models.crnn import CRNN
from tqdm import tqdm
import sys

sys.path.append(".")

# -------------------------------
# Hyperparameters & Config
# -------------------------------
IMAGE_DIR = 'data/images'
LABELS_FILE = 'data/labels.txt'
IMG_HEIGHT = 32
MAX_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
SAVE_PATH = 'checkpoints/best_model.pth'

# Character Set
alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,() "
char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}  # +1 for blank at index 0
idx_to_char = {i: c for c, i in char_to_idx.items()}
num_classes = len(char_to_idx) + 1  # +1 for blank

# -------------------------------
# Dataset & Dataloaders
# -------------------------------
dataset = OCRDataset(
    labels_file=LABELS_FILE,
    image_dir=IMAGE_DIR,
    char_to_idx=char_to_idx,
    img_height=IMG_HEIGHT,
    max_width=MAX_WIDTH,
)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=IMG_HEIGHT, num_classes=num_classes).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# Collate & Padding Helper
# -------------------------------
def collate_batch(batch):
    images, labels, label_lengths = zip(*batch)

    # Pad labels
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    # Stack images
    images = torch.stack(images)
    return images, labels_concat, label_lengths

# -------------------------------
# Train & Validate Loops
# -------------------------------
def train_epoch():
    model.train()
    epoch_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        images, labels_concat, label_lengths = collate_batch(batch)
        images = images.to(device)
        labels_concat = labels_concat.to(device)

        # Forward pass
        outputs = model(images)  # (T, B, C)
        T, B, C = outputs.size()

        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)

        # CTC Loss
        loss = criterion(outputs.log_softmax(2), labels_concat, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)

def validate_epoch():
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images, labels_concat, label_lengths = collate_batch(batch)
            images = images.to(device)
            labels_concat = labels_concat.to(device)

            outputs = model(images)
            T, B, C = outputs.size()
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)

            loss = criterion(outputs.log_softmax(2), labels_concat, input_lengths, label_lengths)
            val_loss += loss.item()

    return val_loss / len(val_loader)

# -------------------------------
# Training Loop
# -------------------------------
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_epoch()
    val_loss = validate_epoch()

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Saved best model to {SAVE_PATH}")

print("Training complete.")
