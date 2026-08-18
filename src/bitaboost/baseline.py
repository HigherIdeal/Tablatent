from __future__ import annotations

import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from .config import resolve_path
from .ensemble import build_final, build_mixed, build_safe_core
from .features import AUX_NAMES, prepare
from .legacy import activate
from .metrics import brier, summary
from .runtime import log, stage


def _params(cfg, loss):
    p=dict(cfg["catboost"]); p["loss_function"]=loss; p["task_type"]="GPU"; p["devices"]=str(cfg["runtime"]["catboost_device"]); p.pop("verbose",None); p["logging_level"]="Silent"
    if cfg["runtime"].get("gpu_ram_part") is not None: p["gpu_ram_part"]=float(cfg["runtime"]["gpu_ram_part"])
    return p


def _prepare_x(frame, features):
    activate(); import run_regime_feature_prediction_suite as regime_core
    return regime_core.prepare_x(frame,features)


def _save(model,out,name,enabled):
    if enabled:
        d=out/"models"; d.mkdir(parents=True,exist_ok=True); model.save_model(str(d/f"{name}.cbm"))


def _joint_mapping(train,keep,class_index,success,nclasses):
    q=np.zeros((2,nclasses),np.float64); gt=train.loc[keep,"game_type"].astype(str).to_numpy(); season=train.loc[keep,"season"].to_numpy(int)
    for gi,dom in enumerate(("R","F")):
        dm=gt==dom
        if dom=="F" and np.any(dm&(season>=2023)): dm &= season>=2023
        fallback=float(success[dm].mean()) if dm.any() else float(success.mean())
        for ci in range(nclasses):
            cm=dm&(class_index==ci); q[gi,ci]=float(success[cm].mean()) if cm.any() else fallback
    return q


def _offset_prior(frame,mean):
    n=pd.to_numeric(frame["asof_pitcher_n"],errors="coerce").to_numpy(float); p=pd.to_numeric(frame["asof_pitcher_success_rate"],errors="coerce").to_numpy(float); b=pd.to_numeric(frame["asof_batter_success_rate"],errors="coerce").to_numpy(float); shr=(n*p+200*mean)/(n+200); shr=np.nan_to_num(shr,nan=mean); b=np.nan_to_num(b,nan=mean)
    recent=frame[["asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate"]].mean(axis=1).to_numpy(float); recent=np.nan_to_num(recent,nan=mean)
    return .65*shr+.25*recent+.10*b


def train(cfg):
    from catboost import CatBoostClassifier,CatBoostRegressor,Pool
    out=resolve_path(cfg,cfg["output"]["dir"]); out.mkdir(parents=True,exist_ok=True); save_models=bool(cfg["output"].get("save_models",False)); data=prepare(cfg); frame=data.frame; tr=frame.loc[data.train_mask].reset_index(drop=True); va=frame.loc[data.valid_mask].reset_index(drop=True); aux=data.aux.loc[data.train_mask].reset_index(drop=True); y=data.y_valid; gt=data.gt_valid; comp={}; timing={}

    t=time.perf_counter()
    with stage("rich models: direct + aux heads + joint"):
        x,cats=_prepare_x(tr,data.feature_sets["rich"]); xv,_=_prepare_x(va,data.feature_sets["rich"]); vp=Pool(xv,cat_features=cats,feature_names=data.feature_sets["rich"]); keep=aux[list(AUX_NAMES)].notna().all(axis=1).to_numpy(); success=tr.loc[keep,"control_success"].to_numpy(np.float32); av=aux.loc[keep,list(AUX_NAMES)].to_numpy(np.float32)
        repeats=int(cfg["recipe"]["direct"]["success_repeats"]); labels=np.column_stack([*[success]*repeats,av]); fw=float(cfg["recipe"]["direct"]["f_weight"]); w=np.where(tr.loc[keep,"game_type"].astype(str).to_numpy()=="F",fw,1.).astype(np.float32); pool=Pool(x.loc[keep],labels,weight=w,cat_features=cats,feature_names=data.feature_sets["rich"]); m=CatBoostRegressor(**_params(cfg,"MultiRMSE")).fit(pool); direct=np.clip(m.predict(vp,ntree_end=int(cfg["recipe"]["direct"]["tree"])),0,1)[:,0].astype(np.float64); comp["direct"]=direct; _save(m,out,"direct_multi",save_models); del m,pool,labels,w; gc.collect()
        for head,key in (("reverse","reverse_tree"),("middle","middle_tree")):
            h=aux[head].notna().to_numpy(); pool=Pool(x.loc[h],aux.loc[h,head].to_numpy(np.int8),cat_features=cats,feature_names=data.feature_sets["rich"]); m=CatBoostClassifier(**_params(cfg,"Logloss")).fit(pool); comp["reverse600" if head=="reverse" else "middle400"]=m.predict_proba(vp,ntree_end=int(cfg["recipe"]["aux_heads"][key]))[:,1].astype(np.float64); _save(m,out,f"aux_{head}",save_models); del m,pool; gc.collect()
        jav=aux.loc[keep,list(AUX_NAMES)].to_numpy(np.int8); codes=jav@(1<<np.arange(len(AUX_NAMES),dtype=np.int16)); classes=np.unique(codes); ci=np.searchsorted(classes,codes); fw=float(cfg["recipe"]["joint"]["f_weight"]); w=np.where(tr.loc[keep,"game_type"].astype(str).to_numpy()=="F",fw,1.).astype(np.float32); pool=Pool(x.loc[keep],ci,weight=w,cat_features=cats,feature_names=data.feature_sets["rich"]); m=CatBoostClassifier(**_params(cfg,"MultiClass")).fit(pool); prob=m.predict_proba(vp,ntree_end=int(cfg["recipe"]["joint"]["tree"])); q=_joint_mapping(tr,keep,ci,success.astype(float),len(classes)); comp["joint"]=np.sum(prob*q[(gt=="F").astype(np.int8)],axis=1); _save(m,out,"joint",save_models); del m,pool,prob,jav,codes,ci,w,x,xv,vp; gc.collect()
    timing["rich_models_sec"]=time.perf_counter()-t

    t=time.perf_counter()
    with stage("gate + conditional"):
        x,cats=_prepare_x(tr,data.feature_sets["hurdle"]); xv,_=_prepare_x(va,data.feature_sets["hurdle"]); vp=Pool(xv,cat_features=cats,feature_names=data.feature_sets["hurdle"]); usable=aux[["reverse","middle"]].notna().all(axis=1).to_numpy(); gate=((aux["reverse"]==0)&(aux["middle"]==0)).to_numpy(); fw=float(cfg["recipe"]["gate_conditional"]["f_weight"]); gw=np.where(tr.loc[usable,"game_type"].astype(str).to_numpy()=="F",fw,1.).astype(np.float32); pool=Pool(x.loc[usable],gate[usable].astype(np.float32),weight=gw,cat_features=cats,feature_names=data.feature_sets["hurdle"]); m=CatBoostRegressor(**_params(cfg,"RMSE")).fit(pool); comp["gate600"]=np.clip(m.predict(vp,ntree_end=int(cfg["recipe"]["gate_conditional"]["gate_tree"])),0,1).astype(np.float64); _save(m,out,"gate_brier",save_models); del m,pool,gw; gc.collect()
        cond=usable&gate; cw=np.where(tr.loc[cond,"game_type"].astype(str).to_numpy()=="F",fw,1.).astype(np.float32); pool=Pool(x.loc[cond],tr.loc[cond,"control_success"].to_numpy(np.int8),weight=cw,cat_features=cats,feature_names=data.feature_sets["hurdle"]); m=CatBoostClassifier(**_params(cfg,"Logloss")).fit(pool); comp["cond400"]=m.predict_proba(vp,ntree_end=int(cfg["recipe"]["gate_conditional"]["conditional_tree"]))[:,1].astype(np.float64); _save(m,out,"conditional",save_models); del m,pool,cw,x,xv,vp; gc.collect()
    timing["gate_conditional_sec"]=time.perf_counter()-t

    mixed,mixed_blend,_,_=build_mixed(y,gt,comp["direct"],comp["reverse600"],comp["middle400"],comp["gate600"],comp["cond400"],cfg["recipe"]["mixed"]); comp["mixed"]=mixed; log(f"[mixed] brier={brier(y,mixed):.12f} R={mixed_blend['R']:.12f} F={mixed_blend['F']:.12f}")

    t=time.perf_counter()
    with stage("offset cross1"):
        x,cats=_prepare_x(tr,data.feature_sets["offset"]); xv,_=_prepare_x(va,data.feature_sets["offset"]); mean=float(tr.control_success.mean()); pt=_offset_prior(tr,mean); pv=_offset_prior(va,mean); residual=tr.control_success.to_numpy(float)-pt; pool=Pool(x,residual,cat_features=cats,feature_names=data.feature_sets["offset"]); vp=Pool(xv,cat_features=cats,feature_names=data.feature_sets["offset"]); m=CatBoostRegressor(**_params(cfg,"RMSE")).fit(pool); comp["offset"]=np.clip(pv+m.predict(vp,ntree_end=int(cfg["recipe"]["offset"]["tree"])),0,1).astype(np.float64); _save(m,out,"offset_cross1",save_models); del m,pool,vp,x,xv,pt,pv,residual; gc.collect()
    timing["offset_sec"]=time.perf_counter()-t

    t=time.perf_counter()
    with stage("structured ids"):
        x,cats=_prepare_x(tr,data.feature_sets["structured"]); xv,_=_prepare_x(va,data.feature_sets["structured"]); keep=aux[["reverse","middle"]].notna().all(axis=1).to_numpy(); r=aux.reverse.to_numpy(); mm=aux.middle.to_numpy(); sy=tr.control_success.to_numpy(); cls=np.where(sy==1,0,np.where((r==0)&(mm==0),1,np.where((r==1)&(mm==0),2,np.where((r==0)&(mm==1),3,4)))).astype(np.int8); pool=Pool(x.loc[keep],cls[keep],cat_features=cats,feature_names=data.feature_sets["structured"]); vp=Pool(xv,cat_features=cats,feature_names=data.feature_sets["structured"]); m=CatBoostClassifier(**_params(cfg,"MultiClass")).fit(pool); comp["structured"]=m.predict_proba(vp,ntree_end=int(cfg["recipe"]["structured"]["tree"]))[:,0].astype(np.float64); _save(m,out,"structured_ids",save_models); del m,pool,vp,x,xv,cls; gc.collect()
    timing["structured_sec"]=time.perf_counter()-t

    safe,sw=build_safe_core(y,gt,comp["mixed"],comp["offset"],comp["joint"]); final,fw=build_final(y,gt,safe,comp["structured"]); comp["safe"]=safe; comp["pred"]=final; metrics={"mixed":summary(y,mixed),"safe":summary(y,safe),"final":summary(y,final),"mixed_blend":mixed_blend,"safe_weights":sw,"final_weights":fw,"timing_sec":timing,"rows":{"train":len(tr),"valid":len(va)}}; ref=cfg["reference"]; metrics["reference_delta"]={"mixed_brier":metrics["mixed"]["brier"]-float(ref["mixed_brier"]),"safe_core_brier":metrics["safe"]["brier"]-float(ref["safe_core_brier"]),"final_brier":metrics["final"]["brier"]-float(ref["final_brier"])}; metrics["reference_pass"]=abs(metrics["final"]["brier"]-float(ref["final_brier"]))<=float(ref["brier_tolerance"])
    if cfg["output"].get("save_components",True): np.savez_compressed(out/"predictions.npz",y=y,gt=gt,**comp,safe_weights_R=np.asarray(sw["R"]),safe_weights_F=np.asarray(sw["F"]),final_weights_R=np.asarray(fw["R"]),final_weights_F=np.asarray(fw["F"]))
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); serial={k:v for k,v in cfg.items() if not k.startswith("_")}; (out/"resolved_config.json").write_text(json.dumps(serial,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); log(f"[final] brier={metrics['final']['brier']:.12f} score={metrics['final']['score']:.4f} reference_pass={metrics['reference_pass']}"); return metrics
