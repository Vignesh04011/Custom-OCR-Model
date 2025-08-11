# decoder.py

def ctc_greedy_decoder(output, idx_to_char, blank=0):
    """
    Converts model output to readable text using greedy decoding.
    
    Args:
        output (Tensor): Output from model, shape (T, B, C)
        idx_to_char (dict): Mapping from class index to character
        blank (int): Index representing the CTC blank token (usually 0)

    Returns:
        List[str]: Decoded text for each item in batch
    """
    pred = output.argmax(2).permute(1, 0)  # shape: (B, T)
    results = []

    for sequence in pred:
        decoded = []
        prev = blank
        for idx in sequence:
            idx = idx.item()
            if idx != blank and idx != prev:
                decoded.append(idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(decoded))

    return results
