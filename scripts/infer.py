"""Inference for one image or a folder; physics targets are never inputs."""

import argparse, json
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from _common import choose_device
from ultralytics.utils import ops
from ur_freezedet.models import URFreezeDet
from ur_freezedet.models.config import load_config
from ur_freezedet.utils.checkpoint import load_checkpoint


def image_tensor(path, size, device):
    image=Image.open(path).convert("RGB").resize((size,size)); value=torch.from_numpy(np.asarray(image,dtype=np.float32).transpose(2,0,1).copy())/255.0
    return value.unsqueeze(0).to(device)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--device",default="auto"); p.add_argument("--image-size",type=int,default=640); p.add_argument("--output",default="predictions.json"); args=p.parse_args()
    device=choose_device(args.device); model=URFreezeDet(load_config(args.config)).to(device); load_checkpoint(args.checkpoint,model,device=device); model.eval(); source=Path(args.input); paths=[source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
    rows=[]
    with torch.no_grad():
        for path in paths:
            out=model(image_tensor(path,args.image_size,device)); detect=out["detect"]
            if isinstance(detect, tuple): detect=detect[0]
            if isinstance(detect, list): detect=detect[0]
            boxes=ops.non_max_suppression(detect, conf_thres=0.25, iou_thres=0.7, nc=1)[0].cpu().tolist()
            rows.append({"image_path":str(path),"tray_boxes":boxes,"probabilities":out["stage_probabilities"][0].cpu().tolist(),"final_class":int(out["stage_probabilities"].argmax(1)[0]),"U_hat":float(out["u_hat"][0]),"Pi_hat":float(out["pi_hat"][0]),"gate_weights":out["gate_weights"][0].cpu().tolist()})
    Path(args.output).write_text(json.dumps(rows,indent=2),encoding="utf-8")
