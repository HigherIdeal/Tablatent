#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/dual_gpu_logs
tablatent_python=/home/kjw/.conda/envs/tablatent/bin/python

CUDA_VISIBLE_DEVICES=2 "$tablatent_python" scripts/run_regime_feature_prediction_suite.py \
  --variants recent_base,full_base,full_fast_cont,full_range_cont,full_both_cont \
  --iterations-grid 300,400,500,600 \
  --task-type GPU --devices 0 --catboost-threads 10 \
  --output-dir outputs/regime_feature_prediction_suite \
  2>&1 | tee outputs/dual_gpu_logs/gpu2_regime_decomposition.log &
gpu2_pid=$!

CUDA_VISIBLE_DEVICES=3 "$tablatent_python" scripts/run_regime_feature_prediction_suite.py \
  --variants recent_base,full_base,full_both_cont,full_both_cont_count_hand,full_both_cont_count_base,full_both_cont_context \
  --iterations-grid 300,400,500,600 \
  --task-type GPU --devices 0 --catboost-threads 10 \
  --output-dir outputs/regime_context_cross_suite \
  2>&1 | tee outputs/dual_gpu_logs/gpu3_context_cross.log &
gpu3_pid=$!

gpu2_status=0
gpu3_status=0
wait "$gpu2_pid" || gpu2_status=$?
wait "$gpu3_pid" || gpu3_status=$?

printf 'GPU2 status: %s\nGPU3 status: %s\n' "$gpu2_status" "$gpu3_status"
test "$gpu2_status" -eq 0
test "$gpu3_status" -eq 0
