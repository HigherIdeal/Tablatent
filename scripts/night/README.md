# Overnight dual-GPU research campaign

This campaign is designed to use roughly one night of wall time without turning the project into blind hyperparameter search. GPU2 and GPU3 explore different hypotheses and both keep the independent-row competition rule.

## Scientific split

### GPU2 — structure / temporal information

GPU2 starts from the EX4/EX5 observation that `ball`, `reverse`, `success`, and `strike` contain direction-invariant pitcher-season information. It asks which parts have marginal pitch-level value after controlling reliability and temporal construction.

It searches:

- trait subsets: `success` plus every combination of `ball/reverse/strike`, with sparse `middle` negative controls;
- frozen history: previous season, previous two seasons, or all previous career seasons;
- empirical reliability shrinkage strengths `k=50/100/200/500/1000`;
- trait-only, context+trait, and current-history+trait feature families;
- current-asof minus frozen-trait deviation features when the matching row-local statistic exists;
- equal, recent-decay, latest-season boost, and short recent-window training weights;
- CatBoost Logloss versus direct RMSE/Brier-oriented regression;
- capacity, L2 and seed refinements only after structural screening.

The runner first evaluates a broad structural grid with cheap models. After the initial queue is exhausted, the best structural families are repeatedly refined until the search deadline. This is intentional successive refinement rather than random HPO.

### GPU3 — rolling OOF calibration / residual stacking

GPU3 rebuilds rolling OOF predictions from the existing `recent_raw_game_type` feature family. It evaluates base variants with full features, IDs removed, or context-only; temporal weighting; and Logloss/RMSE objectives.

For each base OOF family it tests:

- identity;
- temperature scaling;
- intercept-only logit shift;
- affine logit calibration;
- shrinkage domain-affine calibration by prior experience, game type, or prediction confidence;
- residual CatBoost correctors using base logit + frozen stable traits, optionally with context or current history.

A calibration/corrector for fold `s` may use OOF predictions only from folds before `s`. Therefore 2022 is intentionally identity for all post-processing methods, 2023 may learn from 2022, and 2024 may learn from 2022+2023.

The frozen SAFE982 prediction vector is not calibrated directly on 2024 labels: equivalent earlier SAFE OOF vectors are not available, so doing that would be a validation-specific fit rather than a deployment-valid calibration experiment.

## Selection objective

Each trial reports 2022/2023/2024 Brier and AUC. Ranking minimizes:

```text
0.20 * Brier_2022
+ 0.30 * Brier_2023
+ 0.50 * Brier_2024
+ 0.25 * std(Brier_2022, Brier_2023, Brier_2024)
```

This gives the latest regime more relevance while explicitly penalizing one-season-only wins.

## Safety contract

- no evaluation-row sorting, shifts, rolling windows, cumulative aggregation, or cross-row references;
- current official pre-pitch row features are allowed;
- a frozen profile used by a season `s` row contains only seasons `< s`;
- no hidden/evaluation target restoration;
- no SAFE982 tuning against 2024 labels;
- GPU3 stacking is rolling OOF, not in-sample stacking.

## Launch

Activate the existing environment first:

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost

git fetch origin night-campaign-20260818
git switch night-campaign-20260818
git pull origin night-campaign-20260818
```

One-command launch (recommended):

```bash
bash scripts/night/launch_overnight.sh 7.67
```

The launcher isolates physical GPUs 2 and 3 with `CUDA_VISIBLE_DEVICES`; CatBoost sees logical device 0 in each worker. Both workers use `nohup`, stdin is detached, output is unbuffered, and PID files are saved, so closing the terminal does not stop the campaign.

Manual launch in separate terminals is also valid:

```bash
mkdir -p outputs/night_20260819/gpu2
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
python scripts/night/run_gpu2_structure.py \
  --config experiments/configs/night_campaign_20260819.yaml \
  --hours 7.67 --gpu 2 \
> outputs/night_20260819/gpu2/worker.log 2>&1 < /dev/null &
echo $! > outputs/night_20260819/gpu2/pid
```

```bash
mkdir -p outputs/night_20260819/gpu3
nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
python scripts/night/run_gpu3_calibration.py \
  --config experiments/configs/night_campaign_20260819.yaml \
  --hours 7.67 --gpu 3 \
> outputs/night_20260819/gpu3/worker.log 2>&1 < /dev/null &
echo $! > outputs/night_20260819/gpu3/pid
```

For a live combined Markdown report, the one-command launcher also starts a CPU-only summary watcher. If workers are launched manually:

```bash
nohup python scripts/night/watch_summary.py \
  --root outputs/night_20260819 --interval 60 --hours 8.5 \
> outputs/night_20260819/summary_watcher.log 2>&1 < /dev/null &
```

## Monitoring and recovery

Logs:

```bash
tail -f outputs/night_20260819/gpu2/worker.log
tail -f outputs/night_20260819/gpu3/worker.log
```

Refresh/print status:

```bash
python scripts/night/summarize_night.py --root outputs/night_20260819
```

Files updated while running:

```text
outputs/night_20260819/
  overnight_report.md
  campaign_state.json
  gpu2/
    worker.log
    trials.jsonl
    heartbeat.json
    best.json
    best.md
    leaderboard.md
    final_summary.json
  gpu3/
    worker.log
    trials.jsonl
    heartbeat.json
    best.json
    best.md
    base_manifest.json
    base_leaderboard.md
    leaderboard.md
    final_summary.json
```

`trials.jsonl` is append-only and flushed after every trial. Best/heartbeat/leaderboard files are atomically replaced. Re-running the same worker on the same output directory skips previously recorded GPU2 configs; GPU3 reuses saved base OOF artifacts and skips already recorded base+method pairs.

To check whether workers are alive:

```bash
kill -0 $(cat outputs/night_20260819/gpu2/pid) && echo GPU2_ALIVE
kill -0 $(cat outputs/night_20260819/gpu3/pid) && echo GPU3_ALIVE
```

After completion, `overnight_report.md`, each worker's `final_summary.json`, and the two leaderboards are the main files to inspect.
