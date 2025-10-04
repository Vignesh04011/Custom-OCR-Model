"""
scripts/inference_batch.py

Batch inference for answer sheet OCR.
Processes a folder of PDFs and saves extracted question-answer text as JSON.
"""

import os
import json
from tqdm import tqdm
from src.main import process_pdf

def run_inference(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(input_folder, pdf)
        result = process_pdf(pdf_path)

        out_path = os.path.join(output_folder, pdf.replace(".pdf", ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    run_inference("data/pdfs", "data/outputs")
