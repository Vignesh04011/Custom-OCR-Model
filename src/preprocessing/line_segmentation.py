import numpy as np

def segment_lines(block_img):
    """
    Segments a block image into line bounding boxes using horizontal projection.
    :param block_img: binary image (text = white, bg = black)
    :return: list of (y_start, y_end)
    """
    proj = np.sum(block_img, axis=1)
    thresh = np.max(proj) * 0.1

    lines = []
    in_line = False
    start = 0
    for i, val in enumerate(proj):
        if val > thresh and not in_line:
            in_line = True
            start = i
        elif val <= thresh and in_line:
            in_line = False
            lines.append((start, i))
    return lines
