from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def domain_masks(gt):
    gt = np.asarray(gt).astype(str)
    return {d: gt == d for d in ("R", "F")}


def closed_form_domain_blend(y, direct, logic, gt):
    y = np.asarray(y, np.float64); direct = np.asarray(direct, np.float64); logic = np.asarray(logic, np.float64)
    pred = direct.copy(); weights = {}
    for dom, mask in domain_masks(gt).items():
        delta = logic[mask] - direct[mask]
        den = float(np.dot(delta, delta))
        w = 0.0 if den <= 0 else float(np.clip(np.dot(delta, y[mask] - direct[mask]) / den, 0.0, 1.0))
        pred[mask] = direct[mask] + w * delta; weights[dom] = w
    return pred, weights


def fit_simplex(y, matrix, initial=None):
    """Historical R/F simplex optimizer used to create the SAFE artifact."""
    y = np.asarray(y, np.float64); matrix = np.asarray(matrix, np.float64); k = matrix.shape[1]
    x0 = np.full(k, 1.0/k) if initial is None else np.asarray(initial, np.float64)
    x0 = np.clip(x0, 0, 1); x0 /= x0.sum()
    a = matrix.T @ matrix / len(y); v = matrix.T @ y / len(y)
    fun = lambda w: float(w @ a @ w - 2.0 * w @ v)
    jac = lambda w: 2.0 * a @ w - 2.0 * v
    res = minimize(fun, x0, jac=jac, method="SLSQP", bounds=[(0.,1.)]*k,
                   constraints={"type":"eq","fun":lambda w: float(np.sum(w)-1.),"jac":lambda w: np.ones_like(w)},
                   options={"ftol":1e-15,"maxiter":2000})
    if not res.success or not np.isfinite(res.x).all():
        raise RuntimeError(f"simplex optimization failed: {res.message}")
    w = np.clip(np.asarray(res.x, np.float64), 0, 1); w /= w.sum(); return w


def fit_domain_simplex(y, matrix, gt, initial=None):
    y = np.asarray(y, np.float64); matrix = np.asarray(matrix, np.float64); pred = np.empty(len(y)); weights = {}
    for dom, mask in domain_masks(gt).items():
        w = fit_simplex(y[mask], matrix[mask], initial); pred[mask] = matrix[mask] @ w; weights[dom] = w.tolist()
    return pred, weights


def build_mixed(y, gt, direct, reverse600, middle400, learned_gate600, conditional400, cfg):
    c = float(cfg["interaction_c"]); wi = float(cfg["independent_gate_weight"]); wg = float(cfg["learned_gate_weight"])
    if abs(wi + wg - 1.0) > 1e-12: raise ValueError("mixed gate weights must sum to 1")
    independent = np.clip(1.0 - reverse600 - middle400 + c*reverse600*middle400, 0, 1)
    logic = (wi*independent + wg*learned_gate600) * conditional400
    pred, blend = closed_form_domain_blend(y, direct, logic, gt)
    return pred, blend, independent, logic


def build_safe_core(y, gt, mixed, offset, joint):
    return fit_domain_simplex(y, np.column_stack([mixed, offset, joint]), gt, [0.7,0.2,0.1])


def build_final(y, gt, safe, structured):
    return fit_domain_simplex(y, np.column_stack([safe, structured]), gt, [0.98,0.02])
