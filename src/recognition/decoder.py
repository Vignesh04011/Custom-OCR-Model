import torch
import numpy as np

def greedy_decode(output, alphabet, blank=0):
    """
    Greedy CTC decoding.
    :param output: [T,B,C] logit tensor
    :param alphabet: list of characters
    """
    out = torch.argmax(output, dim=2).cpu().numpy().T
    results = []
    for seq in out:
        prev = -1
        s = ""
        for idx in seq:
            if idx != blank and idx != prev:
                s += alphabet[idx]
            prev = idx
        results.append(s)
    return results
