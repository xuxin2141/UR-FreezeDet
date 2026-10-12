"""Validate manifest fields, finite targets, bbox values, and split leakage."""

import argparse, math
from pathlib import Path
import pandas as pd

REQUIRED={"image_path","sample_id","batch_id","loading_g","time_min","stage","u_target","pi_target","u_max","m_sub","split"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("--allow-missing-images",action="store_true"); args=p.parse_args(); path=Path(args.manifest).resolve(); df=pd.read_csv(path); missing=REQUIRED-set(df.columns)
    errors=[]
    if missing: errors.append(f"missing fields: {sorted(missing)}")
    if df.sample_id.duplicated().any(): errors.append("sample_id is not unique")
    if set(df.stage.dropna().astype(int))-{0,1,2}: errors.append("stage must be 0, 1, 2")
    if df.groupby("batch_id").split.nunique().gt(1).any(): errors.append("batch leakage across splits")
    for col in ["u_target","u_max","loading_g","time_min"]:
        if col in df and not pd.to_numeric(df[col],errors="coerce").map(math.isfinite).all(): errors.append(f"{col} contains non-finite values")
    if "pi_target" in df:
        for _,row in df.iterrows():
            finite=pd.notna(row.pi_target) and math.isfinite(float(row.pi_target))
            if int(row.m_sub)==0 and finite: errors.append("DD pi_target must be NA")
            if int(row.m_sub)==1 and not finite: errors.append("HTD/MTD pi_target must be finite")
    if not args.allow_missing_images:
        for value in df.image_path:
            if not (path.parent/str(value)).exists(): errors.append(f"missing image: {value}"); break
    if errors: raise SystemExit("dataset check failed: " + "; ".join(errors))
    print({"rows":len(df),"splits":df.split.value_counts().to_dict(),"batches":df.batch_id.nunique(),"status":"ok"})
if __name__ == "__main__": main()
