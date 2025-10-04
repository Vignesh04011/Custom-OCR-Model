def stitch_pages(pages_answers):
    """
    Merge answers across pages if continuation is detected.
    :param pages_answers: list of dict outputs from question_mapper
    :return: merged answers dict
    """
    final = {}
    last_q = None
    for page_ans in pages_answers:
        for qid, data in page_ans.items():
            if qid in final:
                final[qid]["text"] += " " + data["text"]
                final[qid]["pages"].extend(data["pages"])
                final[qid]["bboxes"].extend(data["bboxes"])
                final[qid]["conf"] = (final[qid]["conf"] + data["conf"]) / 2
            else:
                final[qid] = data
            last_q = qid
    return final
