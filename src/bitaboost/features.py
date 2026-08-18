from __future__ import annotations

import gc
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import resolve_path
from .legacy import activate
from .path_state import add_paths
from .runtime import log, stage

AUX_NAMES = ("reverse", "middle", "ball", "strike")

@dataclass
class PreparedData:
    frame: pd.DataFrame
    train_mask: np.ndarray
    valid_mask: np.ndarray
    y_valid: np.ndarray
    gt_valid: np.ndarray
    aux: pd.DataFrame
    feature_sets: dict[str, list[str]]
    categorical_extra: set[str]


def auxiliary_targets(frame: pd.DataFrame) -> pd.DataFrame:
    rate_columns = {"reverse":"asof_pitcher_reverse_rate","middle":"asof_pitcher_middle_rate","ball":"asof_pitcher_ball_rate","strike":"asof_pitcher_strike_rate"}
    work = frame[["pitcher_id","asof_pitcher_n",*rate_columns.values()]].copy(); work["_index"] = np.arange(len(work),dtype=np.int64)
    work = work.sort_values(["pitcher_id","asof_pitcher_n","_index"],kind="stable").reset_index(drop=True)
    valid_transition = (work.pitcher_id.eq(work.pitcher_id.shift(-1)) & work.asof_pitcher_n.shift(-1).eq(work.asof_pitcher_n+1)).to_numpy()
    n = pd.to_numeric(work.asof_pitcher_n,errors="coerce").to_numpy(np.float64); order=work._index.to_numpy(np.int64)
    out = pd.DataFrame(index=np.arange(len(frame)))
    for short,column in rate_columns.items():
        rate=pd.to_numeric(work[column],errors="coerce").to_numpy(np.float64); count=np.rint(n*rate); delta=np.roll(count,-1)-count
        valid=valid_transition & np.isfinite(delta) & ((delta==0)|(delta==1)); values=np.full(len(work),np.nan,np.float32); values[valid]=delta[valid].astype(np.float32)
        restored=np.full(len(frame),np.nan,np.float32); restored[order]=values; out[short]=restored
    return out


def _legacy_config(cfg: dict) -> dict:
    return {"paths":{"processed_file":str(resolve_path(cfg,cfg["data"]["processed_train"])),"output_dir":str(resolve_path(cfg,cfg["output"]["dir"]))},"data":{"target_col":cfg["data"]["target_col"],"season_col":cfg["data"]["season_col"],"row_id_col":"row_id","train_seasons":[2019,2020,2021,2022],"val_seasons":[2023],"test_seasons":[2024]},"raw_catboost":{"iterations":3000,"learning_rate":.03,"depth":6,"l2_leaf_reg":10.,"random_strength":1.,"early_stopping_rounds":100,"task_type":"GPU","devices":"0","verbose":50,"threshold":.5}}


def _reserve_engineered(frame: pd.DataFrame, context_columns: list[str]) -> pd.DataFrame:
    numeric=["eng_anchor_available","eng_anchor_gap_n"]
    for short in ("success","reverse","middle","ball","strike"):
        numeric += [f"eng_anchor_{short}_rate",f"eng_since_anchor_{short}_rate",f"eng_since_anchor_{short}_minus_long",f"eng_since_anchor_{short}_minus_anchor"]
    numeric += ["eng_banchor_available","eng_banchor_gap_n","eng_banchor_success_rate","eng_banchor_middle_rate","eng_bsince_success_rate","eng_bsince_middle_rate","eng_bsince_success_minus_long","eng_bsince_middle_minus_long","eng_anchor_cross_success","eng_anchor_cross_middle","eng_anchor_pitch_success_shrunk","eng_anchor_batter_success_shrunk","eng_anchor_gap_logratio","eng_anchor_valid_proxy"]
    numeric += [f"eng_anchor_pitch_{s}_shrunk" for s in ("reverse","middle","ball","strike")]
    for p in ("p_hand","b_hand","p_hand_gt","b_hand_gt","p_count","b_count","p_count_hand","b_count_hand","p_pressure","b_pressure","p_inning","b_inning","p_gt","b_gt"):
        numeric += [f"eng_{p}_logn",f"eng_{p}_rate",f"eng_{p}_delta"]
    for p in ("paux_hand","paux_count","paux_count_hand","paux_pressure"):
        numeric.append(f"eng_{p}_logn")
        for s in AUX_NAMES: numeric += [f"eng_{p}_{s}_rate",f"eng_{p}_{s}_delta"]
    for p in ("pcond_hand","pcond_count","pcond_count_hand","pcond_pressure"):
        numeric += [f"eng_{p}_logn",f"eng_{p}_rate",f"eng_{p}_delta"]
    numeric=[c for c in dict.fromkeys(numeric) if c not in frame.columns]
    if numeric:
        block=pd.DataFrame(np.nan,index=frame.index,columns=numeric,dtype=np.float64); frame=pd.concat([frame,block],axis=1,copy=False); del block
    for c in context_columns:
        if c not in frame.columns: frame[c]="<MISSING>"
    return frame


def _add_regime_continuous(train: pd.DataFrame, valid: pd.DataFrame, regime_core) -> None:
    for frame in (train,valid):
        season=pd.to_numeric(frame["season"],errors="raise").astype(int); gt=frame["game_type"].astype("string").fillna("<MISSING>").astype(str).str.upper(); hand=frame["batter_hand"].astype("string").fillna("<MISSING>").astype(str)
        levels=sorted([x for x in hand.unique().tolist() if x!="<MISSING>"])
        if len(levels)<2: raise ValueError(f"expected two batter_hand levels, got {levels}")
        recent=season.ge(2023).to_numpy(); old=~recent; is_r=gt.eq("R").to_numpy(); h1=hand.eq(levels[0]).to_numpy(); h2=hand.eq(levels[1]).to_numpy(); fast=pd.to_numeric(frame["asof_pitcher_fastball_rate"],errors="coerce").to_numpy(np.float32); rrng=pd.to_numeric(frame["eng_ps_recent_range_135"],errors="coerce").to_numpy(np.float32)
        frame[regime_core.RECENT_FLAG]=recent.astype(np.float32)
        for name,mask in {"rr_fastball_hand1":is_r&recent&h1,"rr_fastball_hand2":is_r&recent&h2,"ro_fastball_hand1":is_r&old&h1,"ro_fastball_hand2":is_r&old&h2}.items():
            out=np.full(len(frame),np.nan,np.float32); out[mask]=fast[mask]; frame[name]=out
        rr=np.full(len(frame),np.nan,np.float32); ro=np.full(len(frame),np.nan,np.float32); rr[is_r&recent]=rrng[is_r&recent]; ro[is_r&old]=rrng[is_r&old]; frame["rr_recent_range"]=rr; frame["ro_recent_range"]=ro


def prepare(cfg: dict) -> PreparedData:
    activate()
    import build_recent_regime_submissions as recent_core
    import run_context_interaction_screen as context_core
    import run_frozen_season_anchor_probe as anchor_core
    import run_offset_residual_boosting as offset_core
    import run_regime_feature_prediction_suite as regime_core
    with stage("load + canonical row features"):
        frame,_=recent_core.prepare_frame(_legacy_config(cfg)); frame=frame.reset_index(drop=True); frame["season"]=pd.to_numeric(frame["season"],errors="raise").astype(int); frame["game_type"]=frame["game_type"].astype("string").str.strip().str.upper(); frame=_reserve_engineered(frame,list(context_core.INTERACTION_COLUMNS))
    with stage("training-only auxiliary targets"): aux=auxiliary_targets(frame)
    with stage("frozen historical features"):
        anchor_core.add_frozen_anchor_features(frame,season_col="season",pitcher_col="pitcher_id",n_col="asof_pitcher_n",count_tolerance=.05)
        banchor=offset_core.add_batter_anchor(frame); cross4=offset_core.add_anchor_cross(frame); matchup=offset_core.add_frozen_matchup(frame); count=offset_core.add_frozen_count_profiles(frame); pressure=offset_core.add_frozen_pressure_profiles(frame); domain=offset_core.add_frozen_domain_profiles(frame); auxprof=offset_core.add_frozen_aux_profiles(frame,aux,"pressure"); condprof=offset_core.add_frozen_conditional_profiles(frame,aux,False); context_core.add_context_interactions(frame); paths=add_paths(frame,"pitcher_id","season")+add_paths(frame,"batter_id","season")
    valid_season=int(cfg["data"]["validation_season"]); train_mask=frame["season"].lt(valid_season).to_numpy(); valid_mask=frame["season"].eq(valid_season).to_numpy()
    with stage("regime features"):
        trv=frame.loc[train_mask].copy(); vav=frame.loc[valid_mask].copy(); _add_regime_continuous(trv,vav,regime_core); cols=[regime_core.RECENT_FLAG,*regime_core.FAST_CONT,*regime_core.RANGE_CONT]; frame.loc[train_mask,cols]=trv[cols].to_numpy(); frame.loc[valid_mask,cols]=vav[cols].to_numpy(); del trv,vav
    cross1=["eng_anchor_cross_success","eng_anchor_cross_middle","eng_anchor_pitch_success_shrunk","eng_anchor_batter_success_shrunk","eng_anchor_gap_logratio"]
    anchor_success=["eng_anchor_available","eng_anchor_gap_n","eng_anchor_success_rate","eng_since_anchor_success_rate","eng_since_anchor_success_minus_long"]
    multi_anchor=[x for s in ("reverse","middle","ball","strike") for x in (f"eng_anchor_{s}_rate",f"eng_since_anchor_{s}_rate",f"eng_since_anchor_{s}_minus_long")]
    base=[*recent_core.feature_set("recent_raw_game_type"),regime_core.RECENT_FLAG,*regime_core.FAST_CONT,*regime_core.RANGE_CONT,*context_core.INTERACTION_COLUMNS,*paths]
    rich=[*base,*anchor_success,*multi_anchor,*banchor,*cross4,*matchup,*count,*pressure,*domain,*auxprof]; hurdle=[*rich,*condprof]; offset=[*base,*anchor_success,*banchor,*cross1]
    structured=[*recent_core.feature_set("recent_raw_game_type"),"pitcher_id","batter_id",regime_core.RECENT_FLAG,*regime_core.FAST_CONT,*regime_core.RANGE_CONT,*context_core.INTERACTION_COLUMNS,*paths]
    for name,features in {"rich":rich,"hurdle":hurdle,"offset":offset,"structured":structured}.items():
        if len(features)!=len(set(features)): raise RuntimeError(f"duplicate {name} features")
    path_cats={x for x in paths if x.endswith(("last_gt","current_x_last"))}; extra=set(context_core.INTERACTION_COLUMNS)|path_cats|{"pitcher_id","batter_id"}; regime_core.EXTRA_CATEGORICAL.update(extra)
    frame=frame.copy(); gc.collect(); y=pd.to_numeric(frame.loc[valid_mask,cfg["data"]["target_col"]],errors="raise").to_numpy(np.float64); gt=frame.loc[valid_mask,"game_type"].astype(str).to_numpy(); log(f"[data] train={train_mask.sum():,} valid={valid_mask.sum():,} rich={len(rich)} hurdle={len(hurdle)} offset={len(offset)}")
    return PreparedData(frame,train_mask,valid_mask,y,gt,aux,{"rich":rich,"hurdle":hurdle,"offset":offset,"structured":structured},extra)
