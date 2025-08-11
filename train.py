import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config                     # 📌 Import config module
from dataset import OCRDataset
from models.crnn import CRNN

# -------------------------------
# Dataset
# -------------------------------
dataset = OCRDataset(labels_file=config.LABELS_FILE, image_dir=config.IMAGE_DIR)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def collate_batch(batch):
    images, labels, label_lengths = zip(*batch)
    return torch.stack(images), torch.cat(labels), torch.tensor(label_lengths, dtype=torch.long)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=config.IMG_HEIGHT, num_channels=1, num_classes=config.NUM_CLASSES).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# -------------------------------
# Training Loop
# -------------------------------
best_val_loss = float('inf')
for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

    # Training
    model.train()
    train_loss = 0
    for images, labels_concat, label_lengths in tqdm(train_loader, desc="Training"):
        images, labels_concat = images.to(device), labels_concat.to(device)
        outputs = model(images)
        T, B, C = outputs.size()
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)

        loss = criterion(outputs.log_softmax(2), labels_concat, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels_concat, label_lengths in tqdm(val_loader, desc="Validating"):
            images, labels_concat = images.to(device), labels_concat.to(device)
            outputs = model(images)
            T, B, C = outputs.size()
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)
            loss = criterion(outputs.log_softmax(2), labels_concat, input_lengths, label_lengths)
            val_loss += loss.item()

    print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), config.SAVE_PATH)
        print(f"✅ Saved best model to {config.SAVE_PATH}")

print("Training complete.")
