# EX10 — Domain-shift feature pruning

EX9 showed that 2023 is almost perfectly separable from older seasons by input features (`AUC ≈ 0.979`), while density-ratio reweighting did not improve SAFE982.

EX10 asks a different question: are the features that most strongly reveal the 2023 regime also nuisance variables that hurt 2024 generalization?

The experiment trains a 2023-vs-older domain classifier on <=2023 inputs, ranks its feature importances, and removes the top-k ranked features from only the SAFE direct MultiRMSE head. `game_type` is protected. Reverse/middle, hurdle, offset, joint, structured, and every promoted SAFE mixture/simplex weight remain frozen.

Run:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/ex10/run_domain_feature_pruning.py --gpu 2
```

Outputs:

- `outputs/experiments/ex10_domain_feature_pruning/report.md`
- `outputs/experiments/ex10_domain_feature_pruning/metrics.json`

`drop_k=0` is the exact retrain control and should reproduce SAFE982.
