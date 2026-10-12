# UR-FreezeDet

Algorithm-only release of UR-FreezeDet. The repository contains the shared D1–D7 model, training losses, data interface, and evaluation/inference entrypoints. Real images, annotations, and checkpoints are intentionally distributed separately.

## Variants

| Variant | Added capability |
|---|---|
| D1 | YOLO11n tray detection and flat stage classification |
| D2 | Hierarchical phase/substage classification |
| D3 | D2 plus U regression |
| D4 | D3 plus Pi regression |
| D5 | D4 plus analytical U–Pi consistency |
| D6 | D5 plus learned scale fusion |
| D7 | D5 plus regression-conditioned scale fusion |

## Installation

```bash
pip install -r requirements.txt
```

The locked versions were read from the formal Python environment used by the project. Ultralytics is pinned to 8.3.163.

## Commands

Run from this directory. Supply a private manifest and a released or local pretrained YOLO11n checkpoint through arguments; no data path is embedded in the code.

```bash
python scripts/train.py --config configs/D7.yaml --data /path/to/dataset.yaml
python scripts/validate.py --config configs/D7.yaml --data /path/to/dataset.yaml --checkpoint /path/to/best.pt
python scripts/test.py --config configs/D7.yaml --data /path/to/dataset.yaml --checkpoint /path/to/frozen.pt
python scripts/infer.py --config configs/D7.yaml --checkpoint /path/to/best.pt --input /path/to/image-or-folder
python scripts/train_all.py --data /path/to/dataset.yaml
python scripts/evaluate_all.py --data /path/to/dataset.yaml --checkpoints /path/to/checkpoints.json
```

`train.py` uses FP32, seed 42, 100 epochs by default, validation Macro-F1 followed by validation total loss for best-checkpoint selection, and train/validation splits only. `test.py` requires an explicit frozen checkpoint and never selects one.

## Data contract

The private manifest must provide relative `image_path`, `sample_id`, `batch_id`, `loading_g`, `time_min`, `stage`, `u_target`, `pi_target`, `u_max`, `m_sub`, and `split` fields plus tray bounding-box columns `bbox_cx`, `bbox_cy`, `bbox_w`, and `bbox_h` (or a relative `tray_label_path`). Stage order is HTD=0, MTD=1, DD=2. DD `pi_target` is NA. `u_max` is used only by the training consistency loss.

## Scope and licensing

This release is source code only. Dataset access, released weights, paper result tables, and citation details are maintained separately. Original repository code is released under the MIT License; third-party packages retain their own licenses.
