
def judgement_compare(ai_judgement, gt_judgement):
    # breakpoint()
    ai_value = str(ai_judgement).strip().strip('"').strip("'")
    gt_value = next(
        (key for key, value in gt_judgement.items() if value == "●"),
        None,
    )
    judgement_result = "yes" if ai_value == gt_value else "no"
    # print(judgement_result)
    # breakpoint()
    return judgement_result