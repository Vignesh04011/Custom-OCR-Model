import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.recognition.crnn_model import CRNN
from src.recognition.decoder import greedy_decode
from training.dataloader import OCRDataset, collate_fn, load_vocab

def cer(pred, gt):
    dp = [[i+j if i*j==0 else 0 for j in range(len(pred)+1)] for i in range(len(gt)+1)]
    for i in range(1, len(gt)+1):
        for j in range(1, len(pred)+1):
            cost = 0 if gt[i-1]==pred[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1] / max(1,len(gt))

def validate(model, loader, idx2char, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["imgs"].to(device)
            out = model(imgs)
            preds = greedy_decode(out, idx2char)
            for p, gt in zip(preds, batch["texts"]):
                total += 1
                correct += (cer(p, gt) == 0)
    return correct / max(1, total)

def train(train_manifest, val_manifest, vocab_path, epochs=30, batch_size=8, lr=1e-4, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    char2idx, idx2char = load_vocab(vocab_path)
    idx2char = [idx2char[i] for i in range(len(idx2char))]

    train_ds = OCRDataset(train_manifest, vocab_path)
    val_ds = OCRDataset(val_manifest, vocab_path)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate_fn)

    model = CRNN(nclass=len(char2idx)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        start = time.time()

        for batch in train_dl:
            imgs = batch["imgs"].to(device)
            labels = batch["labels"].to(device)
            input_lens = batch["input_lengths"]
            label_lens = batch["label_lengths"]

            out = model(imgs)
            log_probs = nn.functional.log_softmax(out, dim=2)
            loss = criterion(log_probs, labels, input_lens, label_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = validate(model, val_dl, idx2char, device)
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_dl):.4f} | Val Acc: {acc*100:.2f}% | Time: {time.time()-start:.1f}s")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "char2idx": char2idx
        }, os.path.join(save_dir, f"crnn_epoch{epoch:03d}.pt"))

    print("✅ Training complete!")

if __name__ == "__main__":
    train(
        train_manifest="training/dataset/processed/train.json",
        val_manifest="training/dataset/processed/val.json",
        vocab_path="src/recognition/vocab.txt",
        epochs=40,
        batch_size=8
    )
