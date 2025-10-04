def merge_boxes(boxes, overlap_threshold=0.3):
    """
    Merge overlapping bounding boxes.
    :param boxes: list of [x,y,w,h]
    :return: merged list of boxes
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = []

    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        x, y, w, h = box
        last = merged[-1]
        lx, ly, lw, lh = last

        # compute overlap
        if (x < lx + lw * (1 + overlap_threshold)) and (y < ly + lh * (1 + overlap_threshold)):
            nx = min(x, lx)
            ny = min(y, ly)
            nw = max(x + w, lx + lw) - nx
            nh = max(y + h, ly + lh) - ny
            merged[-1] = [nx, ny, nw, nh]
        else:
            merged.append(box)
    return merged
