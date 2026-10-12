"""JSONL training log helper."""

import json
from pathlib import Path


def append_jsonl(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
