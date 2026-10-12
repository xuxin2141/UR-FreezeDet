"""Train D1 through D7 in order, isolating each output directory."""

import argparse, subprocess, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--output-dir", default="outputs"); p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--device", default="auto"); args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    for i in range(1, 8):
        out = Path(args.output_dir) / f"D{i}"; log = out / "error.log"; out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(root / "scripts" / "train.py"), "--config", str(root / "configs" / f"D{i}.yaml"), "--data", args.data, "--output-dir", str(out), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--device", args.device]
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode: print(f"{i}: failed; see {log}")


if __name__ == "__main__": main()
