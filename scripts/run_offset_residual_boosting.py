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
import run_regime_feature_prediction_suite as regime_core
from src.utils import load_config

def add_batter_anchor(f):
 cols={'success':'asof_batter_success_rate','middle':'asof_batter_middle_rate'};out=['eng_banchor_available','eng_banchor_gap_n',*[f'eng_banchor_{k}_rate' for k in cols],*[f'eng_bsince_{k}_rate' for k in cols],*[f'eng_bsince_{k}_minus_long' for k in cols]]
 for c in out:f[c]=np.nan
 anchor=None
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx];n=part.asof_batter_n.to_numpy(float)
  if anchor is not None:
   ids=part.batter_id;an=ids.map(anchor.asof_batter_n).to_numpy(float);ok=np.isfinite(an)&(n>an);f.loc[idx,'eng_banchor_available']=np.isfinite(an).astype(np.float32);f.loc[idx,'eng_banchor_gap_n']=np.where(ok,n-an,np.nan)
   for k,c in cols.items():
    rate=part[c].to_numpy(float);ar=ids.map(anchor[c]).to_numpy(float);delta=np.rint(n*rate)-np.rint(an*ar);valid=ok&np.isfinite(rate)&np.isfinite(ar)&(delta>=0)&(delta<=n-an);since=np.where(valid,delta/(n-an),np.nan);f.loc[idx,f'eng_banchor_{k}_rate']=ar;f.loc[idx,f'eng_bsince_{k}_rate']=since;f.loc[idx,f'eng_bsince_{k}_minus_long']=since-rate
  else:f.loc[idx,'eng_banchor_available']=0
  latest=part.sort_values(['batter_id','asof_batter_n']).groupby('batter_id',sort=False).tail(1).set_index('batter_id')[['asof_batter_n',*cols.values()]]
  anchor=latest if anchor is None else pd.concat([anchor,latest]).reset_index().sort_values(['batter_id','asof_batter_n']).groupby('batter_id',sort=False).tail(1).set_index('batter_id')
 return out

def add_anchor_cross(f):
 pg=np.nan_to_num(f.eng_anchor_gap_n.to_numpy(float),nan=0);bg=np.nan_to_num(f.eng_banchor_gap_n.to_numpy(float),nan=0);pl=f.asof_pitcher_success_rate.to_numpy(float);bl=f.asof_batter_success_rate.to_numpy(float);ps=np.nan_to_num(f.eng_since_anchor_success_rate.to_numpy(float),nan=pl);bs=np.nan_to_num(f.eng_bsince_success_rate.to_numpy(float),nan=bl);pm=np.nan_to_num(f.eng_since_anchor_middle_rate.to_numpy(float),nan=f.asof_pitcher_middle_rate.to_numpy(float));bm=np.nan_to_num(f.eng_bsince_middle_rate.to_numpy(float),nan=f.asof_batter_middle_rate.to_numpy(float));p=(pg*ps+100*pl)/(pg+100);b=(bg*bs+100*bl)/(bg+100);f['eng_anchor_cross_success']=p-b;f['eng_anchor_cross_middle']=pm-bm;f['eng_anchor_pitch_success_shrunk']=p;f['eng_anchor_batter_success_shrunk']=b;f['eng_anchor_gap_logratio']=np.log1p(pg)-np.log1p(bg);out=['eng_anchor_cross_success','eng_anchor_cross_middle','eng_anchor_pitch_success_shrunk','eng_anchor_batter_success_shrunk','eng_anchor_gap_logratio'];shrunk={}
 for short in ('reverse','middle','ball','strike'):
  long=f[f'asof_pitcher_{short}_rate'].to_numpy(float);state=np.nan_to_num(f[f'eng_since_anchor_{short}_rate'].to_numpy(float),nan=long);name=f'eng_anchor_pitch_{short}_shrunk';f[name]=(pg*state+100*long)/(pg+100);shrunk[short]=f[name].to_numpy(float);out.append(name)
 f['eng_anchor_valid_proxy']=(1-shrunk['reverse'])*(1-shrunk['middle']);out.append('eng_anchor_valid_proxy');return out

def add_frozen_matchup(f):
 specs={'p_hand':(['pitcher_id','batter_hand'],'asof_pitcher_success_rate'),'b_hand':(['batter_id','pitcher_hand'],'asof_batter_success_rate'),'p_hand_gt':(['pitcher_id','batter_hand','game_type'],'asof_pitcher_success_rate'),'b_hand_gt':(['batter_id','pitcher_hand','game_type'],'asof_batter_success_rate')};out=[];states={k:None for k in specs}
 for name in specs:
  for suffix in ('logn','rate','delta'):c=f'eng_{name}_{suffix}';f[c]=np.nan;out.append(c)
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx]
  for name,(keys,basecol) in specs.items():
   state=states[name]
   if state is not None:
    mi=pd.MultiIndex.from_frame(part[keys]);hit=state.reindex(mi);n=hit['n'].to_numpy(float);sm=(hit['sum'].to_numpy(float)+50*part[basecol].to_numpy(float))/(n+50);f.loc[idx,f'eng_{name}_logn']=np.log1p(n);f.loc[idx,f'eng_{name}_rate']=sm;f.loc[idx,f'eng_{name}_delta']=sm-part[basecol].to_numpy(float)
   now=part.groupby(keys,dropna=False).control_success.agg(['size','sum']).rename(columns={'size':'n'});states[name]=now if state is None else pd.concat([state,now]).groupby(level=list(range(len(keys)))).sum()
 return out

def add_frozen_count_profiles(f):
 specs={'p_count':(['pitcher_id','balls_before','strikes_before'],'asof_pitcher_success_rate'),'b_count':(['batter_id','balls_before','strikes_before'],'asof_batter_success_rate'),'p_count_hand':(['pitcher_id','batter_hand','balls_before','strikes_before'],'asof_pitcher_success_rate'),'b_count_hand':(['batter_id','pitcher_hand','balls_before','strikes_before'],'asof_batter_success_rate')};out=[];states={k:None for k in specs}
 for name in specs:
  for suffix in ('logn','rate','delta'):c=f'eng_{name}_{suffix}';f[c]=np.nan;out.append(c)
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx]
  for name,(keys,basecol) in specs.items():
   state=states[name]
   if state is not None:
    hit=state.reindex(pd.MultiIndex.from_frame(part[keys]));n=hit['n'].to_numpy(float);sm=(hit['sum'].to_numpy(float)+100*part[basecol].to_numpy(float))/(n+100);f.loc[idx,f'eng_{name}_logn']=np.log1p(n);f.loc[idx,f'eng_{name}_rate']=sm;f.loc[idx,f'eng_{name}_delta']=sm-part[basecol].to_numpy(float)
   now=part.groupby(keys,dropna=False).control_success.agg(['size','sum']).rename(columns={'size':'n'});states[name]=now if state is None else pd.concat([state,now]).groupby(level=list(range(len(keys)))).sum()
 return out

def add_frozen_pressure_profiles(f):
 specs={'p_pressure':(['pitcher_id','base_state','outs_before'],'asof_pitcher_success_rate'),'b_pressure':(['batter_id','base_state','outs_before'],'asof_batter_success_rate'),'p_inning':(['pitcher_id','inning'],'asof_pitcher_success_rate'),'b_inning':(['batter_id','inning'],'asof_batter_success_rate')};out=[];states={k:None for k in specs}
 for name in specs:
  for suffix in ('logn','rate','delta'):c=f'eng_{name}_{suffix}';f[c]=np.nan;out.append(c)
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx]
  for name,(keys,basecol) in specs.items():
   state=states[name]
   if state is not None:
    hit=state.reindex(pd.MultiIndex.from_frame(part[keys]));n=hit['n'].to_numpy(float);sm=(hit['sum'].to_numpy(float)+100*part[basecol].to_numpy(float))/(n+100);f.loc[idx,f'eng_{name}_logn']=np.log1p(n);f.loc[idx,f'eng_{name}_rate']=sm;f.loc[idx,f'eng_{name}_delta']=sm-part[basecol].to_numpy(float)
   now=part.groupby(keys,dropna=False).control_success.agg(['size','sum']).rename(columns={'size':'n'});states[name]=now if state is None else pd.concat([state,now]).groupby(level=list(range(len(keys)))).sum()
 return out

def add_frozen_opponent_profiles(f):
 specs={'p_opp':(['pitcher_id','batter_team_id'],'asof_pitcher_success_rate'),'b_opp':(['batter_id','pitcher_team_id'],'asof_batter_success_rate'),'p_home':(['pitcher_id','top_bottom'],'asof_pitcher_success_rate'),'b_home':(['batter_id','top_bottom'],'asof_batter_success_rate')};out=[];states={k:None for k in specs}
 for name in specs:
  for suffix in ('logn','rate','delta'):c=f'eng_{name}_{suffix}';f[c]=np.nan;out.append(c)
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx]
  for name,(keys,basecol) in specs.items():
   state=states[name]
   if state is not None:
    hit=state.reindex(pd.MultiIndex.from_frame(part[keys]));n=hit['n'].to_numpy(float);sm=(hit['sum'].to_numpy(float)+100*part[basecol].to_numpy(float))/(n+100);f.loc[idx,f'eng_{name}_logn']=np.log1p(n);f.loc[idx,f'eng_{name}_rate']=sm;f.loc[idx,f'eng_{name}_delta']=sm-part[basecol].to_numpy(float)
   now=part.groupby(keys,dropna=False).control_success.agg(['size','sum']).rename(columns={'size':'n'});states[name]=now if state is None else pd.concat([state,now]).groupby(level=list(range(len(keys)))).sum()
 return out

def add_frozen_domain_profiles(f):
 specs={'p_gt':(['pitcher_id','game_type'],'asof_pitcher_success_rate'),'b_gt':(['batter_id','game_type'],'asof_batter_success_rate')};out=[];states={k:None for k in specs}
 for name in specs:
  for suffix in ('logn','rate','delta'):c=f'eng_{name}_{suffix}';f[c]=np.nan;out.append(c)
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx]
  for name,(keys,basecol) in specs.items():
   state=states[name]
   if state is not None:
    hit=state.reindex(pd.MultiIndex.from_frame(part[keys]));n=hit['n'].to_numpy(float);sm=(hit['sum'].to_numpy(float)+100*part[basecol].to_numpy(float))/(n+100);f.loc[idx,f'eng_{name}_logn']=np.log1p(n);f.loc[idx,f'eng_{name}_rate']=sm;f.loc[idx,f'eng_{name}_delta']=sm-part[basecol].to_numpy(float)
   now=part.groupby(keys,dropna=False).control_success.agg(['size','sum']).rename(columns={'size':'n'});states[name]=now if state is None else pd.concat([state,now]).groupby(level=[0,1]).sum()
 return out

def add_frozen_pair_profile(f):
 out=['eng_pair_logn','eng_pair_rate','eng_pair_delta'];f[out]=np.nan;state=None
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx];keys=['pitcher_id','batter_id']
  if state is not None:
   hit=state.reindex(pd.MultiIndex.from_frame(part[keys]));n=hit['n'].to_numpy(float);base=(.7*part.asof_pitcher_success_rate+.3*part.asof_batter_success_rate).to_numpy(float);sm=(hit['sum'].to_numpy(float)+100*base)/(n+100);f.loc[idx,'eng_pair_logn']=np.log1p(n);f.loc[idx,'eng_pair_rate']=sm;f.loc[idx,'eng_pair_delta']=sm-base
  now=part.groupby(keys).control_success.agg(['size','sum']).rename(columns={'size':'n'});state=now if state is None else pd.concat([state,now]).groupby(level=[0,1]).sum()
 return out

def add_frozen_leverage_profiles(f):
 f['_score_bucket']=pd.to_numeric(f.score_diff_pitcher_team).clip(-3,3).astype(int);f['_li_bucket']=pd.cut(pd.to_numeric(f.li),[-np.inf,.5,1,2,4,np.inf],labels=False).fillna(-1).astype(int);specs={'p_score':(['pitcher_id','_score_bucket'],'asof_pitcher_success_rate'),'b_score':(['batter_id','_score_bucket'],'asof_batter_success_rate'),'p_li':(['pitcher_id','_li_bucket'],'asof_pitcher_success_rate'),'b_li':(['batter_id','_li_bucket'],'asof_batter_success_rate')};out=[];states={k:None for k in specs}
 for name in specs:
  for suffix in ('logn','rate','delta'):c=f'eng_{name}_{suffix}';f[c]=np.nan;out.append(c)
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx]
  for name,(keys,basecol) in specs.items():
   state=states[name]
   if state is not None:
    hit=state.reindex(pd.MultiIndex.from_frame(part[keys]));n=hit['n'].to_numpy(float);sm=(hit['sum'].to_numpy(float)+100*part[basecol].to_numpy(float))/(n+100);f.loc[idx,f'eng_{name}_logn']=np.log1p(n);f.loc[idx,f'eng_{name}_rate']=sm;f.loc[idx,f'eng_{name}_delta']=sm-part[basecol].to_numpy(float)
   now=part.groupby(keys).control_success.agg(['size','sum']).rename(columns={'size':'n'});states[name]=now if state is None else pd.concat([state,now]).groupby(level=[0,1]).sum()
 return out

def add_anchor_form(f):
 out=[]
 for short in ('success','middle'):
  state=f[f'eng_since_anchor_{short}_rate']
  for w in (1,3,5):
   name=f'eng_season_{short}_minus_prev{w}';f[name]=state-f[f'asof_pitcher_prev{w}_game_{short}_rate'];out.append(name)
 return out

def add_arsenal_context(f):
 rates=f[['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']].to_numpy(float);dom=np.array(['F','B','O'])[np.nanargmax(np.nan_to_num(rates,nan=-1),axis=1)];count=f.balls_before.astype(str)+'-'+f.strikes_before.astype(str);hand=f.batter_hand.astype(str);f['eng_arsenal_dom']=dom;f['eng_arsenal_count']=pd.Series(dom,index=f.index)+'|'+count;f['eng_arsenal_hand']=pd.Series(dom,index=f.index)+'|'+hand;f['eng_arsenal_count_hand']=pd.Series(dom,index=f.index)+'|'+count+'|'+hand;return ['eng_arsenal_dom','eng_arsenal_count','eng_arsenal_hand','eng_arsenal_count_hand']

def add_frozen_aux_profiles(f,aux,extra='both'):
 specs={'paux_hand':['pitcher_id','batter_hand'],'paux_count':['pitcher_id','balls_before','strikes_before'],'paux_count_hand':['pitcher_id','batter_hand','balls_before','strikes_before']}
 if extra in ('gt','both'):specs['paux_gt']=['pitcher_id','game_type']
 if extra in ('pressure','both'):specs['paux_pressure']=['pitcher_id','base_state','outs_before']
 if extra=='inning':specs['paux_inning']=['pitcher_id','inning']
 shorts=('reverse','middle','ball','strike');out=[];states={(name,short):None for name in specs for short in shorts}
 for name in specs:
  c=f'eng_{name}_logn';f[c]=np.nan;out.append(c)
  for short in shorts:
   for suffix in ('rate','delta'):c=f'eng_{name}_{short}_{suffix}';f[c]=np.nan;out.append(c)
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx]
  for name,keys in specs.items():
   mi=pd.MultiIndex.from_frame(part[keys])
   for short in shorts:
    state=states[(name,short)];base=part[f'asof_pitcher_{short}_rate'].to_numpy(float)
    if state is not None:
     hit=state.reindex(mi);n=hit['n'].to_numpy(float);sm=(hit['sum'].to_numpy(float)+100*base)/(n+100);f.loc[idx,f'eng_{name}_logn']=np.log1p(n);f.loc[idx,f'eng_{name}_{short}_rate']=sm;f.loc[idx,f'eng_{name}_{short}_delta']=sm-base
    work=part[keys].copy();work['_y']=aux.loc[idx,short].to_numpy();now=work.dropna(subset=['_y']).groupby(keys)._y.agg(['size','sum']).rename(columns={'size':'n'});states[(name,short)]=now if state is None else pd.concat([state,now]).groupby(level=list(range(len(keys)))).sum()
 return out

def add_frozen_conditional_profiles(f,aux,batter=False):
 specs={'pcond_hand':['pitcher_id','batter_hand'],'pcond_count':['pitcher_id','balls_before','strikes_before'],'pcond_count_hand':['pitcher_id','batter_hand','balls_before','strikes_before'],'pcond_pressure':['pitcher_id','base_state','outs_before']}
 if batter:specs.update({'bcond_hand':['batter_id','pitcher_hand'],'bcond_count':['batter_id','balls_before','strikes_before'],'bcond_count_hand':['batter_id','pitcher_hand','balls_before','strikes_before'],'bcond_pressure':['batter_id','base_state','outs_before']})
 out=[];states={k:None for k in specs};hist_n=hist_y=0.
 for name in specs:
  for suffix in ('logn','rate','delta'):c=f'eng_{name}_{suffix}';f[c]=np.nan;out.append(c)
 for season in sorted(f.season.unique()):
  idx=f.index[f.season.eq(season)];part=f.loc[idx];prior=hist_y/hist_n if hist_n else .8
  for name,keys in specs.items():
   state=states[name]
   if state is not None:
    hit=state.reindex(pd.MultiIndex.from_frame(part[keys]));n=hit['n'].to_numpy(float);sm=(hit['sum'].to_numpy(float)+100*prior)/(n+100);f.loc[idx,f'eng_{name}_logn']=np.log1p(n);f.loc[idx,f'eng_{name}_rate']=sm;f.loc[idx,f'eng_{name}_delta']=sm-prior
   work=part[keys+['control_success']].copy();valid=(aux.loc[idx,'reverse'].to_numpy()==0)&(aux.loc[idx,'middle'].to_numpy()==0);work=work.loc[valid];now=work.groupby(keys).control_success.agg(['size','sum']).rename(columns={'size':'n'});states[name]=now if state is None else pd.concat([state,now]).groupby(level=list(range(len(keys)))).sum()
  valid=(aux.loc[idx,'reverse'].to_numpy()==0)&(aux.loc[idx,'middle'].to_numpy()==0);hist_n+=valid.sum();hist_y+=part.control_success.to_numpy()[valid].sum()
 return out

def prior(frame,name,mean):
 if name=='flat':return np.full(len(frame),mean)
 n=frame.asof_pitcher_n.to_numpy(float);p=frame.asof_pitcher_success_rate.to_numpy(float);b=frame.asof_batter_success_rate.to_numpy(float);shr=(n*p+200*mean)/(n+200)
 shr=np.nan_to_num(shr,nan=mean);b=np.nan_to_num(b,nan=mean)
 if name=='pitch':return shr
 if name=='both':return .75*shr+.25*b
 r=frame[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']].mean(axis=1).to_numpy(float)
 r=np.nan_to_num(r,nan=mean)
 if name=='domain':
  is_f=frame.game_type.astype(str).str.upper().eq('F').to_numpy();wr=np.where(is_f,.4,.3);wb=np.where(is_f,.1,.2)
  return (1-wr-wb)*shr+wr*r+wb*b
 base=.65*shr+.25*r+.10*b
 if name=='season':
  gap=np.nan_to_num(frame.eng_anchor_gap_n.to_numpy(float),nan=0);state=np.nan_to_num(frame.eng_since_anchor_success_rate.to_numpy(float),nan=base);k=np.where(frame.game_type.astype(str).str.upper().eq('F'),25.,100.)
  return (gap*state+k*base)/(gap+k)
 return base
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--devices',default='0');ap.add_argument('--fold',type=int,default=2024);ap.add_argument('--priors',default='pitch,both,recent');ap.add_argument('--old-f-weight',type=float,default=1.0);ap.add_argument('--history',choices=['full','recent'],default='full');ap.add_argument('--anchor',choices=['none','success','multi','both','cross'],default='none');ap.add_argument('--matchup',action='store_true');ap.add_argument('--count-profile',action='store_true');ap.add_argument('--aux-profile',action='store_true');a=ap.parse_args();from catboost import CatBoostRegressor,Pool
 f,_=recent_core.prepare_frame(load_config(ROOT/'configs/default.yaml'));f.season=pd.to_numeric(f.season).astype(int);f.game_type=f.game_type.astype('string').str.upper();anchor_core.add_frozen_anchor_features(f,season_col='season',pitcher_col='pitcher_id',n_col='asof_pitcher_n',count_tolerance=.05);banchor=add_batter_anchor(f) if a.anchor in ('both','cross') else [];across=add_anchor_cross(f) if a.anchor=='cross' else [];matchup=add_frozen_matchup(f) if a.matchup else [];count_profile=add_frozen_count_profiles(f) if a.count_profile else []
 if a.aux_profile:
  from run_multitask_outcome_boosting import auxiliary_targets
  aux_profile=add_frozen_aux_profiles(f,auxiliary_targets(f),'pressure')
 else:aux_profile=[]
 context_core.add_context_interactions(f);paths=path_core.add_paths(f,'pitcher_id','season','control_success')+path_core.add_paths(f,'batter_id','season','control_success');paths=[x for x in paths if not x.endswith('_rate')];regime_core.EXTRA_CATEGORICAL.update(x for x in paths if x.endswith(('last_gt','current_x_last')))
 tr,va=f[f.season<a.fold].copy(),f[f.season==a.fold].copy();regime_core.add_regime_features(tr,va,season_col='season',recent_start=2023);tr=tr[tr.season.eq(a.fold-1)].copy() if a.history=='recent' else tr;base=[*recent_core.feature_set('recent_raw_game_type'),regime_core.RECENT_FLAG,*regime_core.FAST_CONT,*regime_core.RANGE_CONT,*context_core.INTERACTION_COLUMNS,*paths];anchor_success=['eng_anchor_available','eng_anchor_gap_n','eng_anchor_success_rate','eng_since_anchor_success_rate','eng_since_anchor_success_minus_long'];anchor_multi=[*anchor_success,*[x for short in ('reverse','middle','ball','strike') for x in (f'eng_anchor_{short}_rate',f'eng_since_anchor_{short}_rate',f'eng_since_anchor_{short}_minus_long')]];extra=[] if a.anchor=='none' else anchor_success+banchor+across if a.anchor in ('both','cross') else anchor_success if a.anchor=='success' else anchor_multi;feats=base+extra+matchup+count_profile+aux_profile;x,c=regime_core.prepare_x(tr,feats);v,_=regime_core.prepare_x(va,feats);yy=va.control_success.to_numpy(float);mean=tr.control_success.mean();vp=Pool(v,cat_features=c,feature_names=feats);out=ROOT/f"outputs/offset_residual_f{a.fold}_{a.history}_{a.anchor}_{a.priors.replace(',','-')}_of{a.old_f_weight:g}{'_match' if a.matchup else ''}{'_count' if a.count_profile else ''}{'_auxprof' if a.aux_profile else ''}";out.mkdir(parents=True,exist_ok=True);saved={'y':yy,'gt':va.game_type.astype(str).to_numpy()};weights=np.where(tr.game_type.astype(str).eq('F')&(tr.season<2023),a.old_f_weight,1.).astype(np.float32)
 for name in a.priors.split(','):
  pt=prior(tr,name,mean);pv=prior(va,name,mean);model=CatBoostRegressor(iterations=600,learning_rate=.03,depth=8,loss_function='RMSE',l2_leaf_reg=20,random_strength=.5,bootstrap_type='Bayesian',bagging_temperature=.5,border_count=128,random_seed=42,has_time=True,one_hot_max_size=10,allow_writing_files=False,task_type='GPU',devices=a.devices,verbose=False).fit(Pool(x,tr.control_success.to_numpy()-pt,weight=weights,cat_features=c,feature_names=feats))
  for t in (200,400,600):
   p=np.clip(pv+model.predict(vp,ntree_end=t),0,1);saved[f'{name}{t}']=p;q=metric_core.binary_metrics(yy,p);print(f'{name}{t}: s={q["score"]:.1f} b={q["brier"]:.3e}')
 np.savez_compressed(out/'predictions.npz',**saved)
if __name__=='__main__':main()
