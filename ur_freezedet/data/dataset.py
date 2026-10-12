from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _median_iqr(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)), "iqr": float(max(np.percentile(values, 75) - np.percentile(values, 25), 1e-6))}


def fit_train_scalers(manifest):
    train = manifest.loc[manifest.split == "train"]
    pi = np.log1p(train.loc[train.m_sub == 1, "pi_target"].dropna().to_numpy())
    return {"u": _median_iqr(train.u_target), "log_pi": _median_iqr(pi), "fit_splits": ["train"]}


class URFreezeDataset(Dataset):
    def __init__(self, manifest_path, split, image_size=640, scalers=None, allow_missing=False):
        self.manifest_path = Path(manifest_path).resolve()
        self.df = pd.read_csv(self.manifest_path)
        self.df = self.df.loc[self.df.split == split].reset_index(drop=True)
        self.image_size, self.scalers = int(image_size), scalers
        self.allow_missing = allow_missing

    def __len__(self): return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = Path(str(row.image_path))
        if not image_path.is_absolute(): image_path = self.manifest_path.parent / image_path
        if not image_path.exists():
            if not self.allow_missing: raise FileNotFoundError(f"image_path does not exist: {image_path}")
            image = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        else:
            image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            box=None,
            reducing_gap=None,
        )
        image = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
        if "tray_label_path" in row.index and pd.notna(row.tray_label_path):
            label_path = Path(str(row.tray_label_path))
            if not label_path.is_absolute(): label_path = self.manifest_path.parent / label_path
            fields = label_path.read_text(encoding="utf-8").split()
            bbox = [float(v) for v in fields[1:]]
        else:
            bbox = [float(row[f"bbox_{axis}"]) for axis in ("cx", "cy", "w", "h")]
        return {
            "image": image, "bbox": torch.tensor(bbox, dtype=torch.float32),
            "stage": int(row.stage), "phase": float(row.phase), "substage": int(row.substage),
            "m_sub": float(row.m_sub), "u_target": float(row.u_target),
            "pi_target": float(row.pi_target) if pd.notna(row.pi_target) else float("nan"), "umax": float(row.umax), "filename": row.sample_id,
        }


def collate_batch(items):
    batch = {"image": torch.stack([x["image"] for x in items]), "filename": [x["filename"] for x in items]}
    for key, dtype in [("stage", torch.long), ("phase", torch.float32), ("substage", torch.long), ("m_sub", torch.float32), ("u_target", torch.float32), ("pi_target", torch.float32), ("umax", torch.float32)]:
        batch[key] = torch.tensor([x[key] for x in items], dtype=dtype)
    batch["det"] = {"batch_idx": torch.arange(len(items)), "cls": torch.zeros(len(items)), "bboxes": torch.stack([x["bbox"] for x in items])}
    return batch
