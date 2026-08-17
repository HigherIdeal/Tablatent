from __future__ import annotations

import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_game_type_temporal_regime_ablation as metric_core
import run_multitask_outcome_boosting as multi_core
import run_regime_feature_prediction_suite as regime_core
import run_frozen_season_anchor_probe as anchor_core
import run_offset_residual_boosting as offset_core
from src.utils import load_config

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,default=2024);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--devices',default='0');ap.add_argument('--ids',action='store_true');ap.add_argument('--anchors',action='store_true');a=ap.parse_args()
 from catboost import CatBoostClassifier,Pool
 f,_=recent_core.prepare_frame(load_config(ROOT/'configs/default.yaml'));f.season=pd.to_numeric(f.season).astype(int);f.game_type=f.game_type.astype('string').str.upper();
 if a.anchors:anchor_core.add_frozen_anchor_features(f,season_col='season',pitcher_col='pitcher_id',n_col='asof_pitcher_n',count_tolerance=.05);banchor=offset_core.add_batter_anchor(f)
 else:banchor=[]
 context_core.add_context_interactions(f)
 paths=path_core.add_paths(f,'pitcher_id','season','control_success')+path_core.add_paths(f,'batter_id','season','control_success');paths=[x for x in paths if not x.endswith('_rate')];regime_core.EXTRA_CATEGORICAL.update(x for x in paths if x.endswith(('last_gt','current_x_last')))
 tr,va=f[f.season<a.fold].copy(),f[f.season==a.fold].copy();regime_core.add_regime_features(tr,va,season_col='season',recent_start=2023)
 ids=['pitcher_id','batter_id'] if a.ids else [];regime_core.EXTRA_CATEGORICAL.update(ids)
 anchors=[] if not a.anchors else ['eng_anchor_available','eng_anchor_gap_n',*[x for short in ('success','reverse','middle','ball','strike') for x in (f'eng_anchor_{short}_rate',f'eng_since_anchor_{short}_rate',f'eng_since_anchor_{short}_minus_long')],*banchor]
 feats=[*recent_core.feature_set('recent_raw_game_type'),*ids,regime_core.RECENT_FLAG,*regime_core.FAST_CONT,*regime_core.RANGE_CONT,*context_core.INTERACTION_COLUMNS,*paths,*anchors]
 aux=multi_core.auxiliary_targets(tr);keep=aux[['reverse','middle']].notna().all(axis=1).to_numpy();r=aux.reverse.to_numpy();m=aux.middle.to_numpy();y=tr.control_success.to_numpy();cls=np.where(y==1,0,np.where((r==0)&(m==0),1,np.where((r==1)&(m==0),2,np.where((r==0)&(m==1),3,4)))).astype(np.int8)
 x,c=regime_core.prepare_x(tr.loc[keep],feats);v,_=regime_core.prepare_x(va,feats);vp=Pool(v,cat_features=c,feature_names=feats)
 model=CatBoostClassifier(iterations=600,learning_rate=.03,depth=8,loss_function='MultiClass',l2_leaf_reg=20,random_strength=.5,bootstrap_type='Bayesian',bagging_temperature=.5,border_count=128,random_seed=a.seed,has_time=True,one_hot_max_size=10,allow_writing_files=False,task_type='GPU',devices=a.devices,verbose=False).fit(Pool(x,cls[keep],cat_features=c,feature_names=feats))
 yy=va.control_success.to_numpy(float);out=ROOT/f"outputs/structured_multiclass_f{a.fold}_s{a.seed}{'_ids' if a.ids else ''}{'_anchors' if a.anchors else ''}";out.mkdir(parents=True,exist_ok=True);saved={'y':yy,'gt':va.game_type.astype(str).to_numpy()}
 for t in (200,400,600):
  p=model.predict_proba(vp,ntree_end=t)[:,0];saved[f't{t}']=p;q=metric_core.binary_metrics(yy,p);print(f"t{t}: s={q['score']:.1f} b={q['brier']:.3e}")
 np.savez_compressed(out/'predictions.npz',**saved)
if __name__=='__main__':main()
