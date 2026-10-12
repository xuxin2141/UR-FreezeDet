"""Evaluate D1-D7 from explicit frozen checkpoint paths."""

import argparse, json, subprocess, sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--checkpoints",required=True,help="JSON mapping D1-D7 to checkpoint paths"); p.add_argument("--device",default="auto"); args=p.parse_args(); mapping=json.loads(Path(args.checkpoints).read_text(encoding="utf-8")); root=Path(__file__).resolve().parents[1]; results={}
    for i in range(1,8):
        name=f"D{i}"; command=[sys.executable,str(root/"scripts"/"test.py"),"--config",str(root/"configs"/(name+".yaml")),"--data",args.data,"--checkpoint",mapping[name],"--device",args.device]; result=subprocess.run(command,cwd=root,capture_output=True,text=True); results[name]={"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
    print(json.dumps(results,indent=2))
if __name__ == "__main__": main()
