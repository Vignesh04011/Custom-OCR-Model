import cv2

def adaptive_binarize(img):
    """
    Adaptive binarization for variable lighting.
    :param img: np.ndarray (BGR or Gray)
    :return: binary image (inverted: text = white, bg = black)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 9
    )
    return binary
