import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        tokens = [t.strip() for t in f if t.strip()]
    char2idx = {c: i for i, c in enumerate(tokens)}
    idx2char = {i: c for c, i in enumerate(tokens)}
    return char2idx, idx2char

def text_to_labels(text, char2idx):
    return [char2idx[c] for c in text if c in char2idx]

def resize_keep_ratio(img, height=32, max_width=2048):
    h, w = img.shape[:2]
    scale = height / h
    new_w = min(int(w * scale), max_width)
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

def normalize(img):
    img = img.astype(np.float32) / 255.0
    return (img - 0.5) / 0.5

class OCRDataset(Dataset):
    def __init__(self, manifest_path, vocab_path, img_height=32, transform=None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.char2idx, _ = load_vocab(vocab_path)
        self.img_height = img_height
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(s["img_path"])
        img = resize_keep_ratio(img, self.img_height)
        if self.transform:
            img = self.transform(img)
        img = normalize(img)
        label = text_to_labels(s["text"], self.char2idx)
        return {"img": img, "label": label, "text": s["text"]}

def collate_fn(batch):
    heights = [b["img"].shape[0] for b in batch]
    assert len(set(heights)) == 1, "All images must have same height"
    H = heights[0]
    widths = [b["img"].shape[1] for b in batch]
    max_w = max(widths)

    imgs = np.zeros((len(batch), 1, H, max_w), np.float32)
    labels = []
    label_lens, input_lens = [], []

    for i, b in enumerate(batch):
        w = b["img"].shape[1]
        imgs[i, 0, :, :w] = b["img"]
        labels.extend(b["label"])
        label_lens.append(len(b["label"]))
        input_lens.append(max_w // 4)  # CRNN stride≈4

    return {
        "imgs": torch.tensor(imgs),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_lengths": torch.tensor(label_lens, dtype=torch.long),
        "input_lengths": torch.tensor(input_lens, dtype=torch.long),
        "texts": [b["text"] for b in batch],
    }
