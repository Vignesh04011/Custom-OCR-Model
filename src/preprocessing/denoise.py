import cv2

def denoise(img):
    """
    Apply fast non-local means denoising.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
