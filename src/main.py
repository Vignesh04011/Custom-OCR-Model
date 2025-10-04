import cv2
import torch
import json
from src.preprocessing.deskew import deskew_image
from src.preprocessing.binarize import adaptive_binarize
from src.preprocessing.line_segmentation import segment_lines
from src.detection.contour_detector import detect_blocks
from src.mapping.question_mapper import map_lines_to_qids
from src.mapping.continuation_handler import stitch_pages
from src.integration.pdf_handler import pdf_to_images
from src.recognition.crnn_model import CRNN
from src.recognition.decoder import greedy_decode

def load_model(model_path, vocab_path, device="cpu"):
    """Load trained CRNN model + vocab."""
    checkpoint = torch.load(model_path, map_location=device)
    char2idx = checkpoint["char2idx"]
    idx2char = [c for _, c in sorted(char2idx.items(), key=lambda x: x[1])]

    model = CRNN(nclass=len(char2idx)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, idx2char

def recognize_line(img, model, idx2char, device="cpu"):
    """Run CRNN on a single line image."""
    h, w = img.shape
    img = cv2.resize(img, (int(w * (32 / h)), 32))
    img = (img.astype("float32") / 255.0 - 0.5) / 0.5
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        preds = greedy_decode(out, idx2char)
    return preds[0]

def process_pdf(pdf_path, model_path="models/crnn_best.pt", vocab_path="src/recognition/vocab.txt", device=None):
    """Full OCR pipeline for answer sheet PDF."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, idx2char = load_model(model_path, vocab_path, device)

    images = pdf_to_images(pdf_path)
    pages_data = []

    for page_no, img_bytes in enumerate(images):
        np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        img = deskew_image(gray)
        bin_img = adaptive_binarize(img)
        blocks = detect_blocks(bin_img)

        lines_out = []
        for box in blocks:
            x, y, w, h = box
            block = bin_img[y:y+h, x:x+w]
            line_boxes = segment_lines(block)
            for (y0, y1) in line_boxes:
                line_img = block[y0:y1, :]
                text = recognize_line(line_img, model, idx2char, device)
                lines_out.append({
                    "page": page_no + 1,
                    "bbox": [x, y + y0, w, y1 - y0],
                    "text": text,
                    "conf": 1.0
                })

        qmap = map_lines_to_qids(lines_out)
        pages_data.append(qmap)

    results = stitch_pages(pages_data)
    return results

if __name__ == "__main__":
    out = process_pdf("data/sample.pdf")
    print(json.dumps(out, indent=2, ensure_ascii=False))
