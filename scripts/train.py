"""Train one D1-D7 model using train/validation splits only."""

import argparse, json, shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
from _common import ROOT, choose_device, environment_info, write_json
from ur_freezedet.data import URFreezeDataset, collate_batch
from ur_freezedet.losses import MultiTaskLoss
from ur_freezedet.losses.detection import TrayDetectionLoss
from ur_freezedet.models import URFreezeDet
from ur_freezedet.models.config import load_config
from ur_freezedet.utils.checkpoint import save_checkpoint, load_checkpoint
from ur_freezedet.utils.logging import append_jsonl
from ur_freezedet.utils.seed import seed_everything


def move_batch(batch, device):
    return {k: ({sk: sv.to(device) for sk, sv in v.items()} if k == "det" else v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_epoch(model, loader, criterion, det_criterion, config, device, optimizer=None):
    model.train(optimizer is not None); total = 0.0; all_pred = []; all_true = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(batch["image"]); det_loss, _ = det_criterion(outputs["detect"], batch["det"])
            terms = criterion(outputs, batch, det_loss, config); loss = terms["total"]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        all_pred.extend(outputs["stage_probabilities"].argmax(1).detach().cpu().tolist()); all_true.extend(batch["stage"].cpu().tolist())
        total += float(loss.detach())
    from ur_freezedet.metrics import classification_metrics
    metrics = classification_metrics(all_true, all_pred) if all_true else {"macro_f1": 0.0}
    metrics["total_loss"] = total / max(1, len(loader)); return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0); parser.add_argument("--output-dir", default="outputs/D7")
    parser.add_argument("--resume"); parser.add_argument("--pretrained"); parser.add_argument("--precision", choices=["fp32"], default="fp32")
    args = parser.parse_args(); seed_everything(args.seed); device = choose_device(args.device)
    config = load_config(args.config); data_cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    manifest = Path(args.data).parent / data_cfg["manifest"]; output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train_ds = URFreezeDataset(manifest, "train", allow_missing=False); val_ds = URFreezeDataset(manifest, "val", allow_missing=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_batch)
    model = URFreezeDet(config, pretrained=Path(args.pretrained) if args.pretrained else False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if args.resume: load_checkpoint(args.resume, model, optimizer, device=device)
    criterion = MultiTaskLoss(); det_criterion = TrayDetectionLoss(model.yolo); best_key = None
    shutil.copy2(args.config, output / "config.yaml"); shutil.copy2(args.data, output / "dataset.yaml"); write_json(output / "environment.json", environment_info())
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, criterion, det_criterion, config, device, optimizer); val = run_epoch(model, val_loader, criterion, det_criterion, config, device)
        record = {"epoch": epoch, "train": train, "validation": val}; append_jsonl(output / "training_log.jsonl", record)
        key = (-val["macro_f1"], val["total_loss"])
        if best_key is None or key < best_key:
            best_key = key; save_checkpoint(output / "best.pt", model, optimizer, {"epoch": epoch, "validation": val, "variant": config["variant"]})
        if epoch == args.epochs: save_checkpoint(output / "last.pt", model, optimizer, {"epoch": epoch, "validation": val, "variant": config["variant"]})


if __name__ == "__main__": main()
