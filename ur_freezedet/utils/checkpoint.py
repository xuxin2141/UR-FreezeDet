"""Checkpoint save/load helpers with explicit paths."""

from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer=None, metadata=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "metadata": metadata or {}}, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    payload = torch.load(str(path), map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    if optimizer is not None and payload.get("optimizer") is not None: optimizer.load_state_dict(payload["optimizer"])
    return payload.get("metadata", {})
