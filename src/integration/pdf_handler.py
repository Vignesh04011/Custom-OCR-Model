import fitz  # PyMuPDF

def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF to list of page images (RGB).
    """
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = pix.tobytes("png")
        images.append(img)
    return images
