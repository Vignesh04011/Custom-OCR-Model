import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from config import IMG_HEIGHT, MAX_WIDTH, char_to_idx

class OCRDataset(Dataset):
    def __init__(self, labels_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(labels_file, 'r', encoding='utf-8') as f:
            self.data = [line.strip().split('|') for line in f.readlines()]

        if self.transform is None:
            self.transform = T.Compose([
                T.RandomRotation(2),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])

    def __len__(self):
        return len(self.data)

    def encode_label(self, text):
        return [char_to_idx[c] for c in text if c in char_to_idx]

    def __getitem__(self, idx):
        image_name, label_text = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("L")
        w, h = image.size
        new_w = min(int(w * (IMG_HEIGHT / h)), MAX_WIDTH)
        image = image.resize((new_w, IMG_HEIGHT), Image.BILINEAR)

        padded_image = Image.new('L', (MAX_WIDTH, IMG_HEIGHT), color=255)
        padded_image.paste(image, (0, 0))

        image_tensor = self.transform(padded_image)
        label_indices = self.encode_label(label_text)
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return image_tensor, label_tensor, len(label_indices)
