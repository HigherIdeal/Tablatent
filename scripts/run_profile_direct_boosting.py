from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_frozen_season_anchor_probe as anchor_core
import run_game_type_temporal_regime_ablation as metric_core
import run_offset_residual_boosting as profile_core
import run_regime_feature_prediction_suite as regime_core
from src.utils import load_config
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--devices',default='0');a=ap.parse_args();from catboost import CatBoostClassifier,Pool
 f,_=recent_core.prepare_frame(load_config(ROOT/'configs/default.yaml'));f.season=pd.to_numeric(f.season).astype(int);f.game_type=f.game_type.astype('string').str.upper();anchor_core.add_frozen_anchor_features(f,season_col='season',pitcher_col='pitcher_id',n_col='asof_pitcher_n',count_tolerance=.05);rich=['eng_anchor_available','eng_anchor_gap_n',*[x for short in ('success','reverse','middle','ball','strike') for x in (f'eng_anchor_{short}_rate',f'eng_since_anchor_{short}_rate',f'eng_since_anchor_{short}_minus_long')]];rich+=profile_core.add_batter_anchor(f);rich+=profile_core.add_anchor_cross(f);rich+=profile_core.add_frozen_matchup(f);rich+=profile_core.add_frozen_count_profiles(f);rich+=profile_core.add_frozen_pressure_profiles(f);rich+=profile_core.add_frozen_domain_profiles(f);context_core.add_context_interactions(f);paths=path_core.add_paths(f,'pitcher_id','season','control_success')+path_core.add_paths(f,'batter_id','season','control_success');paths=[x for x in paths if not x.endswith('_rate')];regime_core.EXTRA_CATEGORICAL.update(x for x in paths if x.endswith(('last_gt','current_x_last')))
 tr,va=f[f.season<2024].copy(),f[f.season==2024].copy();regime_core.add_regime_features(tr,va,season_col='season',recent_start=2023);feats=[*recent_core.feature_set('recent_raw_game_type'),regime_core.RECENT_FLAG,*regime_core.FAST_CONT,*regime_core.RANGE_CONT,*context_core.INTERACTION_COLUMNS,*paths,*rich];x,c=regime_core.prepare_x(tr,feats);v,_=regime_core.prepare_x(va,feats);vp=Pool(v,cat_features=c,feature_names=feats);model=CatBoostClassifier(iterations=800,learning_rate=.03,depth=8,loss_function='Logloss',l2_leaf_reg=20,random_strength=.5,bootstrap_type='Bayesian',bagging_temperature=.5,border_count=128,random_seed=42,has_time=True,one_hot_max_size=10,allow_writing_files=False,task_type='GPU',devices=a.devices,verbose=False).fit(Pool(x,tr.control_success,cat_features=c,feature_names=feats));y=va.control_success.to_numpy(float);out=ROOT/'outputs/profile_direct';out.mkdir(parents=True,exist_ok=True);saved={'y':y,'gt':va.game_type.astype(str).to_numpy()}
 for t in (200,400,600,800):p=model.predict_proba(vp,ntree_end=t)[:,1];saved[f't{t}']=p;q=metric_core.binary_metrics(y,p);print(f't{t}: s={q["score"]:.1f} b={q["brier"]:.3e}')
 np.savez_compressed(out/'predictions.npz',**saved)
if __name__=='__main__':main()
