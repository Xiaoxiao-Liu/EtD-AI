# utils/io.py
import json
from pathlib import Path

def read_etd_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_output(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
