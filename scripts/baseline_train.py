from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from bitaboost.config import load_config
from bitaboost.runtime import configure_cuda,configure_warnings

def main():
    ap=argparse.ArgumentParser(description="Train the recovered rule-safe 981.489 baseline on one RTX 4090."); ap.add_argument("--config",default="configs/baseline_safe_981.yaml"); a=ap.parse_args(); cfg=load_config(a.config); configure_cuda(cfg); configure_warnings(cfg)
    from bitaboost.baseline import train
    train(cfg)
if __name__=="__main__": main()
