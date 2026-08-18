import numpy as np
from bitaboost.ensemble import build_final,build_mixed,build_safe_core,closed_form_domain_blend

def test_closed_form_blend_never_worse_than_direct_per_domain():
    rng=np.random.default_rng(1); y=rng.integers(0,2,1000).astype(float); direct=rng.uniform(.3,.7,1000); logic=rng.uniform(.2,.8,1000); gt=np.where(np.arange(1000)%2,"R","F"); p,w=closed_form_domain_blend(y,direct,logic,gt)
    for dom in ("R","F"):
        m=gt==dom; assert np.mean((p[m]-y[m])**2)<=np.mean((direct[m]-y[m])**2)+1e-14; assert 0<=w[dom]<=1

def test_domain_simplex_contracts():
    rng=np.random.default_rng(2); y=rng.integers(0,2,300).astype(float); gt=np.where(np.arange(300)%3,"R","F"); a,b,c=(rng.random(300) for _ in range(3)); safe,sw=build_safe_core(y,gt,a,b,c); final,fw=build_final(y,gt,safe,rng.random(300)); assert np.isfinite(final).all()
    for weights in (sw,fw):
        for dom in ("R","F"):
            w=np.asarray(weights[dom]); assert np.all(w>=-1e-12); assert abs(w.sum()-1)<1e-10

def test_recovered_mixed_formula():
    y=np.array([0.,1.,0.,1.]); gt=np.array(["R","R","F","F"]); direct=np.array([.4,.6,.4,.6]); r=np.array([.1,.2,.1,.2]); m=np.array([.2,.1,.2,.1]); gate=np.array([.7,.8,.7,.8]); cond=np.array([.6,.7,.6,.7]); cfg={"interaction_c":-1.2,"independent_gate_weight":.4,"learned_gate_weight":.6}; pred,w,ind,logic=build_mixed(y,gt,direct,r,m,gate,cond,cfg); expected=np.clip(1-r-m-1.2*r*m,0,1); assert np.allclose(ind,expected); assert np.allclose(logic,(.4*expected+.6*gate)*cond); assert np.isfinite(pred).all(); assert set(w)=={"R","F"}
