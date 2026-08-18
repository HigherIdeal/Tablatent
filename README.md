# Bitaboost

Compact research repository for the LG Aimers 9 baseball next-pitch control-probability task.

The frozen baseline in `configs/baseline_safe_981.yaml` reconstructs the rule-safe research lineage whose 2024 validation checkpoint was:

```text
Brier = 0.247355098397
Score = 981.4893
```

It is intentionally **not** a submission builder. Training/evaluation and future research are kept separate from eventual ZIP packaging.

## Structure

```text
configs/
  baseline_safe_981.yaml       # frozen baseline recipe + runtime policy
scripts/
  prepare_data.py              # CSV -> one fast pickle cache
  baseline_train.py            # baseline train + 2024 evaluation
  eval.py                      # evaluate saved predictions only
src/bitaboost/
  baseline.py                  # compact training orchestration
  features.py                  # exact SAFE feature lineage
  ensemble.py                  # recovered R/F blend/simplex logic
  runtime.py                   # one-GPU + quiet logging policy
  _legacy/                     # quarantined historical feature definitions
experiments/
  configs/                     # future experiment configs
outputs/
  baseline/                    # one baseline artifact set
  experiments/                 # isolated research outputs
```

The `_legacy` directory is not an experiment surface. It preserves only the historical feature definitions needed to reproduce the discovered baseline while the public `scripts/` surface stays small.

## Run

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost

python scripts/prepare_data.py --config configs/baseline_safe_981.yaml
python scripts/baseline_train.py --config configs/baseline_safe_981.yaml
python scripts/eval.py --config configs/baseline_safe_981.yaml
```

The baseline runner pins `CUDA_VISIBLE_DEVICES=2` before CatBoost is imported. CatBoost therefore sees exactly one logical GPU and uses `devices="0"`. Multi-GPU settings are rejected by config validation.

## Training behavior

The runtime is deliberately quiet and memory-conscious:

- CatBoost uses `logging_level=Silent` and `allow_writing_files=false`.
- Only pandas `PerformanceWarning` is suppressed; numerical/runtime errors are not hidden.
- Full feature engineering happens once.
- Large engineered columns are reserved before mutation and the DataFrame is consolidated once before CatBoost phases.
- Related models reuse one transformed feature matrix within a phase.
- Rich, hurdle, offset, and structured feature matrices are processed sequentially, not held simultaneously.
- Models/Pools are released between fits to control host/GPU memory.
- The duplicate historical conditional fit is trained once because the two original branches used identical data/features/loss/weights/seed.
- Only stage duration, key Brier values, blend weights, and final reference status are printed.

CatBoost does not expose a conventional neural-network minibatch-size knob for this workflow. The speed policy therefore focuses on feeding large Pools efficiently and removing CPU/DataFrame/I/O stalls rather than inventing a fake batch-size control.

## Recovered SAFE composition

`mixed`:

```text
direct = MultiRMSE success head @ 600
independent_gate = clip(1 - reverse600 - middle400 - 1.2*reverse600*middle400)
hybrid_gate = 0.4*independent_gate + 0.6*brier_gate600
logic = hybrid_gate * conditional400
mixed = domain-wise closed-form blend(direct, logic)
```

Then:

```text
SAFE = R/F simplex(mixed, old-cross1 offset400, joint600)
FINAL = R/F simplex(SAFE, pre-rich structured-id600)
```

Recovered reference weights are kept in config as audit expectations. Training refits the domain weights on the 2024 validation predictions instead of relying on rounded constants.

## Research policy

Do not create `scripts/run_foo_v2_final2.py` for every idea. Prefer a config under `experiments/configs/` and reusable code under `src/bitaboost/`. An experiment gets a dedicated runner only when it truly has a different workflow.

Every evaluation row remains an independent prediction target. Hidden-test peer aggregation, hidden-test rolling state, and test-distribution adaptation are outside the SAFE baseline contract.
