"""
Prepare IAM Handwriting dataset (line-level) using HuggingFace mirror.

This will:
- Download IAM-line dataset
- Save images locally under training/dataset/raw/
- Create JSON manifests in training/dataset/processed/
"""

from datasets import load_dataset
import os, json
from tqdm import tqdm

RAW_DIR = "training/dataset/raw/"
PROC_DIR = "training/dataset/processed/"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

def save_split(split, name):
    ds = load_dataset("Teklia/IAM-line", split=split)
    manifest = []
    for i, sample in enumerate(tqdm(ds, desc=f"Saving {name}")):
        img = sample["image"]
        text = sample["text"]

        # save image
        img_path = os.path.join(RAW_DIR, f"{name}_{i:06d}.png")
        img.save(img_path)

        manifest.append({"img_path": img_path, "text": text})

    out_json = os.path.join(PROC_DIR, f"{name}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(manifest)} samples to {out_json}")

if __name__ == "__main__":
    save_split("train", "train")
    save_split("validation", "val")
    save_split("test", "test")
    print("✅ IAM dataset prepared.")
