import torch
from torchvision import transforms
from PIL import Image

import config                        # 📌 Import config
from models.crnn import CRNN
from utils.decoder import ctc_greedy_decoder

def preprocess_image(img_path):
    image = Image.open(img_path).convert('L')
    w, h = image.size
    new_w = min(int(w * (config.IMG_HEIGHT / h)), config.MAX_WIDTH)

    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, new_w)),
        transforms.Pad((0, 0, config.MAX_WIDTH - new_w, 0), fill=255),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image)

# -------------------------------
# Load Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=config.IMG_HEIGHT, num_channels=1, num_classes=config.NUM_CLASSES).to(device)
model.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
model.eval()

# -------------------------------
# Inference
# -------------------------------
img_path = "data/images/sample_handwriting.png"
image_tensor = preprocess_image(img_path).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    decoded_texts = ctc_greedy_decoder(output, config.idx_to_char)

print("📝 Predicted Text:", decoded_texts[0])
