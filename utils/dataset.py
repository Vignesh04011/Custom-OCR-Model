import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class OCRDataset(Dataset):
    def __init__(self, labels_file, image_dir, char_to_idx, img_height=32, max_width=128, transform=None):
        """
        labels_file: path to labels.txt (format: img001.png|Q1)
        image_dir: path to folder with images
        char_to_idx: dictionary mapping characters to indices
        img_height: height to which all images are resized
        max_width: width to pad images (optional)
        transform: torchvision transforms (resize, normalize, etc.)
        """
        self.image_dir = image_dir
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.max_width = max_width
        self.transform = transform

        # Read image-label pairs
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.data = [line.strip().split('|') for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def encode_label(self, text):
        # Convert string to list of indices
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def __getitem__(self, idx):
        image_name, label_text = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load grayscale image
        image = Image.open(image_path).convert("L")

        # Resize to fixed height, adjust width proportionally
        w, h = image.size
        new_w = int(self.img_height * w / h)
        new_w = min(new_w, self.max_width)  # prevent very wide images
        image = image.resize((new_w, self.img_height), Image.BILINEAR)

        # Padding to max_width
        padded_image = Image.new('L', (self.max_width, self.img_height), color=255)
        padded_image.paste(image, (0, 0))

        # Transform to tensor and normalize
        if self.transform:
            image_tensor = self.transform(padded_image)
        else:
            transform = T.Compose([
                T.ToTensor(),             # Converts to [0, 1]
                T.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
            ])
            image_tensor = transform(padded_image)

        label_indices = self.encode_label(label_text)
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return image_tensor, label_tensor, len(label_indices)
