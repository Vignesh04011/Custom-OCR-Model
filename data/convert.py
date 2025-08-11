import os
from pdf2image import convert_from_path
import cv2
import uuid

# Paths
PDF_PATH = "1-3 2014.pdf"      # PDF with handwritten text
OUTPUT_IMG_DIR = "data/images"  # Where cropped images will be saved
LABELS_FILE = "data/labels.txt" # Labels file

# Create folders
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Step 1: Convert PDF to images
print("[INFO] Converting PDF to images...")
pages = convert_from_path(PDF_PATH, dpi=300)
page_images = []
for i, page in enumerate(pages):
    img_path = f"page_{i+1:03}.png"
    page.save(img_path, "PNG")
    page_images.append(img_path)

# Step 2: Annotation function
def annotate_image(image_path):
    global ref_point, cropping, clone, img, filename_list, labels_list
    img = cv2.imread(image_path)
    clone = img.copy()
    filename_list = []
    labels_list = []
    cv2.namedWindow("Annotate")
    cv2.setMouseCallback("Annotate", click_and_crop)

    while True:
        cv2.imshow("Annotate", img)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset
        if key == ord("r"):
            img = clone.copy()

        # Press 'q' to quit annotation
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return filename_list, labels_list

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, img, filename_list, labels_list
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(img, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Annotate", img)

        # Save cropped region
        roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        if roi.size > 0:
            filename = f"{uuid.uuid4().hex[:8]}.png"
            save_path = os.path.join(OUTPUT_IMG_DIR, filename)
            cv2.imwrite(save_path, roi)

            # Ask for label
            label = input(f"Enter text for {filename}: ")
            filename_list.append(filename)
            labels_list.append(label)

# Step 3: Loop through PDF pages
with open(LABELS_FILE, "a", encoding="utf-8") as f:
    for page_img in page_images:
        print(f"[INFO] Annotating {page_img}...")
        filename_list, labels_list = annotate_image(page_img)
        for fn, lbl in zip(filename_list, labels_list):
            f.write(f"{fn}|{lbl}\n")

print("[INFO] Annotation complete. Data saved to", LABELS_FILE)
