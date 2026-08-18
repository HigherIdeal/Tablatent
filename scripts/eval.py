from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from bitaboost.config import load_config,resolve_path
from bitaboost.metrics import summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/baseline_safe_981.yaml"); ap.add_argument("--predictions",default=None); a=ap.parse_args(); cfg=load_config(a.config); path=Path(a.predictions) if a.predictions else resolve_path(cfg,cfg["output"]["dir"])/"predictions.npz"; z=np.load(path,allow_pickle=True); y=z["y"].astype(float); gt=z["gt"].astype(str); p=z["pred"].astype(float); result={"all":summary(y,p)}
    for dom in ("R","F"):
        m=gt==dom; result[dom]={"rows":int(m.sum()),**summary(y[m],p[m])}
    ref=float(cfg["reference"]["final_brier"]); result["reference"]={"brier":ref,"delta":result["all"]["brier"]-ref,"pass":abs(result["all"]["brier"]-ref)<=float(cfg["reference"]["brier_tolerance"])}; print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
