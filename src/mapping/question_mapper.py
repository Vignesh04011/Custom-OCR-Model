import re

def map_lines_to_qids(lines):
    """
    Group recognized lines into questions.
    :param lines: list of dict {page, bbox, text, conf}
    :return: dict {QID: {"text":..., "pages":[], "bboxes":[], "conf":avg_conf}}
    """
    answers = {}
    current_q = None

    for line in lines:
        text = line["text"].strip()
        conf = line["conf"]

        match = re.match(r'^(?:Q[\s:.-]*)?(\d+)[\.\)\:]?', text, re.I)
        if match:
            qid = f"Q{match.group(1)}"
            current_q = qid
            answers[qid] = {
                "text": text[match.end():].strip(),
                "pages": [line["page"]],
                "bboxes": [line["bbox"]],
                "conf": conf
            }
        elif current_q:
            answers[current_q]["text"] += " " + text
            answers[current_q]["pages"].append(line["page"])
            answers[current_q]["bboxes"].append(line["bbox"])
            answers[current_q]["conf"] = (answers[current_q]["conf"] + conf) / 2

    return answers
