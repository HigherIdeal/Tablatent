from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from bitaboost.config import load_config, resolve_path

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/baseline_safe_981.yaml"); ap.add_argument("--force",action="store_true"); a=ap.parse_args(); cfg=load_config(a.config); raw=resolve_path(cfg,cfg["data"]["raw_train"]); out=resolve_path(cfg,cfg["data"]["processed_train"]); manifest=resolve_path(cfg,cfg["data"]["manifest"])
    if out.is_file() and not a.force: print(f"[data] reuse {out}"); return
    if not raw.is_file(): raise FileNotFoundError(raw)
    import pandas as pd
    print(f"[data] read {raw}",flush=True); frame=pd.read_csv(raw,low_memory=False); target=cfg["data"]["target_col"]; season=cfg["data"]["season_col"]
    if target not in frame or season not in frame: raise ValueError("train.csv is missing target/season")
    y=pd.to_numeric(frame[target],errors="raise"); years=sorted(pd.to_numeric(frame[season],errors="raise").astype(int).unique().tolist())
    if y.isna().any() or not set(y.unique()).issubset({0,1}): raise ValueError("control_success must be finite binary labels")
    if years != [2019,2020,2021,2022,2023,2024]: raise ValueError(f"unexpected seasons: {years}")
    out.parent.mkdir(parents=True,exist_ok=True); frame.to_pickle(out); manifest.parent.mkdir(parents=True,exist_ok=True); info={"source":str(raw.relative_to(ROOT) if raw.is_relative_to(ROOT) else raw),"source_sha256":sha256(raw),"processed":str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),"rows":len(frame),"columns":frame.shape[1],"seasons":years,"target_mean":float(y.mean())}; manifest.write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(f"[data] wrote {out} rows={len(frame):,} cols={frame.shape[1]}")
if __name__=="__main__": main()
