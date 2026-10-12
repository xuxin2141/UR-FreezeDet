"""Shared CLI path and device helpers."""

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def choose_device(value):
    import torch
    if value == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def environment_info():
    import torch
    try: import ultralytics; ultra = ultralytics.__version__
    except Exception: ultra = "unavailable"
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: commit = None
    return {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultra, "cuda": torch.cuda.is_available(), "git_commit": commit}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
