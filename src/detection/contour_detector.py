import cv2

def detect_blocks(bin_img):
    """
    Detect text blocks using contours + morphology.
    :param bin_img: binary image (text=white, bg=black)
    :return: list of bounding boxes [x,y,w,h]
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(bin_img, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 500:  # filter tiny noise
            boxes.append([x, y, w, h])

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # top-to-bottom
    return boxes
