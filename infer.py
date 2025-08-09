import torch
from torchvision import transforms
from PIL import Image
from models.crnn import CRNN
from utils.decoder import ctc_greedy_decoder

# Reuse character dictionary from training
alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,() "
idx_to_char = {i + 1: c for i, c in enumerate(alphabet)}  # +1 offset
num_classes = len(idx_to_char) + 1  # +1 for blank

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(img_path, img_height=32, max_width=128):
    image = Image.open(img_path).convert('L')  # grayscale
    w, h = image.size
    new_w = min(int(w * (img_height / h)), max_width)

    transform = transforms.Compose([
        transforms.Resize((img_height, new_w)),
        transforms.Pad((0, 0, max_width - new_w, 0), fill=255),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return transform(image)

# -------------------------------
# Load Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_height=32, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()

# -------------------------------
# Inference
# -------------------------------
img_path = "data/images/sample.png"  # update with your test image
image_tensor = preprocess_image(img_path).unsqueeze(0).to(device)  # shape (1, 1, H, W)

with torch.no_grad():
    output = model(image_tensor)  # (T, B, C)
    decoded_texts = ctc_greedy_decoder(output, idx_to_char)

print("📝 Predicted Text:", decoded_texts[0])
