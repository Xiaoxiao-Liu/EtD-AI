# utils/parsing.py
import re
import sys
from typing import List, Dict



def parse_judgement_options(options: List[str]) -> Dict[str, str]:
    """
    Parse judgement options like:
    ['○ No', '○ Probably no', '● Yes', "○Don't know"]

    Return:
    {'No': '○', 'Probably no': '○', 'Yes': '●', "Don't know": '○'}
    """
    result = {}
    # breakpoint()
    for item in options:
        item = item.strip()

        match = re.match(r'^(\S)\s*(.*)$', item)
        if not match:
            print(f"[ERROR] Unrecognized option format: {repr(item)}")
            sys.exit(1)

        symbol, text = match.groups()
        text = text.strip()

        if not text:
            print(f"[ERROR] Empty judgement text after parsing: {repr(item)}")
            sys.exit(1)

        result[text] = symbol

    # ✅ 关键：返回前强校验
    if not result:
        print("[FATAL] Parsed judgement result is empty. Input options:")
        for opt in options:
            print(f"  - {repr(opt)}")
        sys.exit(1)

    # breakpoint()
    return result
    
def extract_model_output(response) -> None:
    # print(response)
    judgement_match = re.search(
        r'judgement:\s*(.*?)\s*reason:',
        response,
        re.IGNORECASE | re.DOTALL,
    )
    reason_match = re.search(
        r'reason:\s*(.*)$',
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if not judgement_match or not reason_match:
        return None, None
    judgement = judgement_match.group(1).strip().strip('"')
    reason = reason_match.group(1).strip().strip('"')
    return judgement, reason
