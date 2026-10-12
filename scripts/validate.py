"""Run validation only with an explicitly supplied checkpoint."""

import argparse
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
from _common import choose_device
from train import run_epoch
from ur_freezedet.data import URFreezeDataset, collate_batch
from ur_freezedet.losses import MultiTaskLoss
from ur_freezedet.losses.detection import TrayDetectionLoss
from ur_freezedet.models import URFreezeDet
from ur_freezedet.models.config import load_config
from ur_freezedet.utils.checkpoint import load_checkpoint


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--data",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--device",default="auto"); p.add_argument("--batch-size",type=int,default=2); args=p.parse_args()
    device=choose_device(args.device); config=load_config(args.config); d=yaml.safe_load(Path(args.data).read_text(encoding="utf-8")); manifest=Path(args.data).parent/d["manifest"]
    ds=URFreezeDataset(manifest,"val"); loader=DataLoader(ds,args.batch_size,shuffle=False,collate_fn=collate_batch); model=URFreezeDet(config).to(device); load_checkpoint(args.checkpoint,model,device=device)
    print(run_epoch(model,loader,MultiTaskLoss(),TrayDetectionLoss(model.yolo),config,device))
if __name__ == "__main__": main()
