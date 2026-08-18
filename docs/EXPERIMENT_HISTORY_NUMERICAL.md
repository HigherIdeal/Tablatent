# Tablatent Experiment History — Numerical Record

> Snapshot: 2026-08-18  
> Scope: LG Aimers 9 baseball control-success prediction project (`HigherIdeal/Tablatent`)  
> Purpose: preserve **what was tried, the observed numerical result, why it was kept/rejected, and what survived into the current safe model**.

## 0. Reading rules

This document separates experiments into five statuses.

- **KEEP / SOTA**: currently useful and rule-safe.
- **RETAINED-CONCEPT**: the exact experiment was not deployed, but its idea survived in later models.
- **REJECTED**: rule-safe, but numerically inferior, unstable, redundant, or too slow.
- **INVALID-RULE**: numerical result is invalid for submission because hidden-test rows were allowed to influence one another.
- **PENDING / INCOMPLETE**: code or run exists, but no reliable final metric was preserved in the tracked repository/conversation.

### Metric comparability warning

Brier scores are the primary comparable quantity. Competition-style scores depend on the target prior of the evaluated fold, so scores from different folds/proxy definitions are **not directly comparable**. A large score in an old screen does not automatically mean that model is stronger than the current 2024/2025-oriented pipeline.

---

# 1. SOTA lineage and external checkpoints

| Stage | Core idea | Internal Brier / score | External LB | Decision |
|---|---|---:|---:|---|
| Early recent raw CatBoost | recent regime only | local metric varied by fold | **858.16** | useful baseline |
| Gated dual-track + R-fast | full + recent + R specialist | weighted Brier around **0.24789** | **886.21** | promoted |
| Safe decomposed ensemble | multitask + hurdle + offset + joint + frozen profiles + R/F ensemble | **0.247355098 / 981.5** | **1098.86143** | **current safe SOTA** |
| Prefix-state family | test-row prefix reconstruction + multi-scale state | as high as **0.246752060 / 1222.8806** | not valid | **INVALID-RULE** |

The current safe submission was explicitly audited so that predicting the whole hidden-test batch and predicting rows individually produced the same result (`absdiff = 0`). The prefix family failed this independence requirement and is quarantined.

---

# 2. Early latent / neural representation branch

## 2.1 Two-branch VAE Stage 1

**Idea:** compress current context and historical/as-of state into separate 16-D latent vectors, then predict `control_success` from the frozen posterior means.

Representative Stage-1 training trace:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 14 | 0.16636 | 0.04009 |
| 15 | 0.16222 | 0.03892 |

These are VAE reconstruction/regularization losses, not Brier scores.

**Decision:** representation learning itself converged, but predictive Stage-2 results were too weak. The branch is no longer the reference model.

## 2.2 Frozen latent heads

| Head | Validation Brier | BCE | AUC | Score / note | Decision |
|---|---:|---:|---:|---:|---|
| Linear | **0.25032231** | — | 0.52875 | — | rejected |
| Bilinear context-history | **0.25070280** | 0.69469438 | 0.52751 | — | rejected |
| MLP | **0.25137261** | 0.69618374 | 0.52944 | official-style 0 | rejected |
| CatBoost on 32-D latent | **0.24964003** | **0.69242962** | **0.53032** | **143.99** | best latent head, still weak |

CatBoost used only `mu_context(16D) + mu_history(16D)` and reached best iteration 23. It improved over neural heads, but remained materially below the later raw-feature CatBoost family.

**Why rejected:** increasing head capacity did not solve the problem; information loss in the compressed representation was more important than classifier capacity.

## 2.3 kNN / local probability on latent

FAISS cross-fit neighbor probability and Stage-2 local-probability variants were implemented. A representative Stage-2 run produced validation Brier around **0.25138** and failed to beat the local-prior baseline near **0.24996**.

**Decision:** rejected as the primary path; latent neighborhood did not provide stable temporal generalization.

---

# 3. LLM / foundation-model tabular attempts

## 3.1 Qwen3-1.7B RAG

**Idea:** retrieve similar historical rows and ask Qwen3-1.7B to infer binary control probability.

Ultrafast path result:

- throughput: **59.5 rows/s**
- Brier: **0.43175814**
- score: **-72,703.26**

**Why rejected:** severe overconfidence from raw binary logits, context truncation, and inference cost. Numerically unusable relative to CatBoost.

## 3.2 TabPFN

The first run hit memory pressure/OOM; `col_chunk_size=2` allowed progress, but validation had only reached roughly **7,000 / 245,525 rows** when it was judged too slow for practical iteration.

**Decision:** rejected operationally before a trustworthy full-fold metric was produced.

---

# 4. Raw CatBoost and canonical feature policy

## 4.1 Early raw CatBoost reference

A 2023 validation baseline was around Brier **0.2499806** at effectively trivial/best-iteration-0 behavior in one early screen, revealing severe temporal drift and weak learnable signal under the original setup.

A later historical `reference_790` feature pipeline produced:

| Variant | 2023 Brier | 2024 Brier | Mean Brier | Interpretation |
|---|---:|---:|---:|---|
| reference_790 | **0.25352775** | **0.24812560** | **0.250827** | unstable across regime |
| drop batter_id | 0.25332898 | **0.24802218** | 0.250676 | slight improvement |
| drop team IDs | 0.25327418 | 0.24824636 | 0.250760 | mixed |
| drop retained IDs | 0.25319697 | 0.24827931 | 0.250738 | mixed |
| drop game context | **0.25158139** | 0.24882267 | **0.250202** | 2023 gain, 2024 loss |
| drop all engineered | — | 0.24815127 | ~0.250785 | no robust gain |

**Key conclusion:** identity features, especially raw pitcher/batter IDs, did not generalize reliably; the large 2023 anomaly was mostly a regime problem rather than a generic need to delete all context.

## 4.2 Canonical de-duplication ablation

After deterministic redundant columns were removed/normalized, the canonical ablation gave approximately:

- reference mean Brier: **0.250494**
- best `drop_game_phase`: **0.249116**
- mean improvement: **-0.001379 Brier**
- worst fold delta: **+0.000149**
- mean AUC: **0.540730**

Features whose removal hurt:

| Removed feature group | Mean Brier |
|---|---:|
| pitcher profile | **0.250865** |
| pitcher recent form | **0.250846** |
| season | **0.250785** |
| handedness | **0.250693** |

ID add-back controls:

| Added ID | Mean Brier | Decision |
|---|---:|---|
| pitcher_id | **0.250645** | exclude |
| batter_id | **0.250676** | exclude |

**Retained:** team IDs, pitcher long-term/recent profile, handedness, canonical state.  
**Excluded:** pitcher_id, batter_id, deterministic redundant state columns.

---

# 5. `game_type` regime shift and temporal adaptation

## 5.1 Structural break

The most important observed shift was in `game_type=F`:

- 2022 success rate: approximately **0.7087**
- 2023 success rate: approximately **0.4729**
- change: approximately **-0.236 (-23.6 percentage points)**

Regime-atlas analysis ranked `game_type` as a major shift variable with shift score around **0.05778** and changepoint **2023**.

**Conclusion:** `game_type` cannot simply be interpreted with one stationary relationship across 2019-2024.

## 5.2 Temporal lagged `game_type` effects

Variants included raw `game_type`, drop `game_type`, previous-season effect, and 2-season EWMA effect.

**Decision:** lagged-effect engineering did not provide a robust replacement for raw `game_type`; later external tests showed keeping raw `game_type` was better than dropping it.

Exact result CSVs are not tracked in Git, so no unsupported number is inserted here.

## 5.3 Hard training-window experiment

Tested `all / 3y / 2y / 1y` windows with raw vs dropped `game_type`.

The critical numerical finding came from the latest-season contribution check for 2024:

| Training policy | Brier | Score | AUC |
|---|---:|---:|---:|
| 2019-2023, raw `game_type` | **0.247940** | **747** | **0.54873** |
| exclude 2023, raw `game_type` | **0.253812** | 0 | 0.52373 |
| 2019-2023, drop `game_type` | **0.248086** | 689 | — |
| exclude 2023, drop `game_type` | **0.249428** | 152 | — |

**Conclusion:** 2023 data is extremely important for predicting 2024. Old data should not be discarded blindly, but the latest regime must be represented.

## 5.4 Smooth recency weighting

A later half-life screen showed all smooth weighting variants worse than unweighted full history:

| Half-life | Weighted Brier | Delta vs unweighted |
|---:|---:|---:|
| no weighting | **0.24791945** | 0 |
| 60 | 0.24794424 | +0.00002479 |
| 36 | 0.24794656 | +0.00002711 |
| 24 | 0.24797332 | +0.00005387 |
| 12 | 0.24801205 | +0.00009260 |

**Why rejected:** old seasons still reduced variance; continuous down-weighting threw away useful stable conditional structure.

---

# 6. Dual-track and R/F specialist family

## 6.1 Expert screen

Weighted proxy summary:

| Expert | Weighted Brier |
|---|---:|
| full_raw | **0.24792363** |
| stable_drop_gt | 0.24799039 |
| recent_raw | 0.24814370 |
| r_full | 0.24847725 |
| r_both | 0.24852254 |
| r_fast | 0.24863254 |
| r_range | 0.24872297 |

Individual specialists were weaker than the full model, but had useful error diversity.

## 6.2 Gated blend

Reference blend:

```text
p_base = 0.8 * p_full + 0.2 * p_recent
R rows: p = 0.9 * p_base + 0.1 * p_r_fast
F rows: p = p_base
```

Weighted result:

- Brier: **0.24789305**
- raw score: **+741.44**
- `alpha_recent=0.20`
- `beta_r=0.10`

A season-forward 2024 screen gave approximately:

- Brier **0.24790735**
- score **760.42**
- R Brier **0.247976**
- F Brier **0.247399**

A finer dual-track alpha sweep found a regime optimum around:

- grid alpha: **0.920**, Brier **0.24787875**
- analytic alpha: **0.918**

The exact alpha was treated cautiously because it was selected on a proxy fold.

**External progression:** recent_raw **858.16** -> gated dual-track + R-fast **886.21**.

**Decision:** specialist is retained only as a small correction; never deploy the R specialist alone.

---

# 7. Dynamic gate, HMM, and sequence models

## 7.1 Row-wise dynamic expert gate

| Variant | Weighted Brier |
|---|---:|
| outputs-only gate | **0.24792259** |
| fixed gate | 0.24792295 |
| delta | **-0.00000036** |

Only **1/3 folds** improved; row-context strength was effectively zero.

**Decision:** rejected. Gain is noise-scale.

## 7.2 HMM latent regime

Robustness after splitting seasons into independent sequences:

- 2023 overlap with 2024 dominant state: **1.000**
- 2024 purity: **0.875 or 1.000**
- partition ARI mean: **0.457**
- median ARI: **0.572**
- minimum ARI: **-0.034**

**Decision:** HMM confirmed a broad `2023≈2024` recent environment, but exact latent-state partition was unstable. Sequential hidden-test adaptation would also violate row independence. Analysis-only.

## 7.3 Stable Player Dynamics GRU

| Fold | Base | lag1 | GRU | GRU+lag1 |
|---:|---:|---:|---:|---:|
| 2022 | **0.24334657** | 0.24339064 | 0.24336559 | 0.24336860 |
| 2023 | 0.25295847 | 0.25283480 | 0.25284750 | **0.25275885** |
| 2024 | **0.24791014** | 0.24792848 | 0.24793613 | 0.24793463 |

Mean GRU+lag1 Brier: approximately **0.248021**; improvement only **1/3 folds**.

**Why rejected:** it helped mainly at the 2023 regime crossing and worsened 2022/2024 robustness. Sequence-state deployment also conflicts with strict hidden-row independence unless reduced to fixed row-local features.

## 7.4 Tiny Transformer

No weighted improvement over the CatBoost temporal baseline was observed.

**Decision:** rejected; sequence complexity did not justify itself.

---

# 8. R-experience specialization

## 8.1 Hard experience routing

R-only baseline:

- GLOBAL_R Brier: **0.24801791**
- score: **750.77**
- loss: **0.68916706**

7-band routed specialists:

- routed Brier: **0.24831134**
- score: **633.35**
- delta: **+0.00029342**

Selected individual band Briers included:

- P0: 0.27040855
- P1: 0.26344503
- P2: 0.24850137
- P3: 0.24795138
- P4: 0.24853551
- P5: 0.24802122
- P6: 0.24846599

Simpler band counts also degraded:

| Routing | Brier | Score |
|---|---:|---:|
| GLOBAL_R | **0.24801812** | **750.69** |
| 3-band | 0.24816370 | 692.43 |
| 5-band | 0.24830568 | 635.62 |
| 7-band | 0.24832321 | 628.60 |

**Why rejected:** hard experience segmentation increases variance, especially for low-support P0/P1 groups.

## 8.2 Empirical-Bayes experience shrinkage

- GLOBAL_R: **0.24801819 / 750.66**
- best tested EB setting (`EB_200`) was still worse than GLOBAL_R.

**Decision:** rejected as an R routing mechanism. Experience remains useful as a continuous/shrunk context feature, not as a hard expert boundary.

---

# 9. Hierarchical probability / empirical historical profiles

An early hierarchical screen built leakage-safe historical probabilities by count, game type, handedness, base state, and experience with empirical-Bayes smoothing.

One early configuration (`H1_g2_profile`) reported fold scores:

```text
1812.37 / 1456.47 / 1543.79 / 1639.21 / 1376.40
```

with overall score **1575.27** under that experiment's own fold/reference definition. An H2 empirical-Bayes variant showed at least one fold score **1852.94**.

**Important:** these old scores are not directly comparable to later 2024 proxy scores because the fold/reference construction differs.

**Decision:** the *concept* survived. Later safe models use frozen matchup/count/pressure/domain/auxiliary profiles, but the old screen itself is not treated as current SOTA evidence.

---

# 10. Model-family alternatives

## 10.1 LightGBM

A representative 2023 model-family screen produced:

- full-feature Brier: approximately **0.251336**
- `no_season_identity`: approximately **0.250946**

**Decision:** rejected; removing season identity helped somewhat, but the family did not solve future-season generalization and remained below CatBoost.

## 10.2 XGBoost regime probe

Depth 2/3/5/7 variants were executed and artifacts preserved locally, but the exact result CSV is not tracked in the Git snapshot.

**Status:** executed, not promoted; no supported exact metric available in the current archive.

## 10.3 Wide & Deep Brier model

A `wide_deep_brier` run exists and produced prediction artifacts.

**Status:** not promoted; exact metric is not preserved in tracked text sources.

---

# 11. Piecewise / calibration approaches

## 11.1 GBDT-guided piecewise linear calibration

Fold results:

| Fold | raw | quantile | global | dual |
|---:|---:|---:|---:|---:|
| 2022 Brier | **0.24372165** | 0.24384073 | 0.24380553 | 0.24395732 |
| 2023 Brier | 0.25193357 | 0.25200656 | 0.25257355 | **0.25165207** |
| 2024 Brier | 0.24865848 | 0.24863955 | **0.24855562** | 0.24855737 |

Weighted Brier:

- raw: **0.248099**
- quantile: 0.248156
- global: 0.248302
- dual: **0.248050**

**Decision:** dual reduced the 2023 failure but did not reach the stronger specialist/decomposed ensemble. Kept as evidence that domain-specific calibration can help, not deployed as main model.

## 11.2 Previous-season cheatsheet

2024 Brier deltas versus baseline:

- pitcher-rich: **+0.00006050**
- player: **+0.00009553**
- matchup: **+0.00007947**

**Decision:** rejected. Raw previous-season lookup overfits stale state; this motivated relative/decayed variants instead.

---

# 12. Trackman / physical information

## 12.1 Deterministic Trackman linkage audit

This was a data-engineering success even though physical models did not yet beat SOTA.

- exactly matched games: **2,693**
- aligned pitches: **808,856**
- state-sequence mismatches: **0**
- pitchers with mapping evidence: **731**
- high-purity accepted pitchers: **584**
- review candidates: 47
- insufficient-support: 100
- team conflicts: **0**
- Trackman-ID duplicate conflicts: **0**
- accepted mappings cover **800,401 / 808,856 ≈ 98.95%** of aligned pitches

**Decision:** linkage infrastructure retained.

## 12.2 Pitch Arsenal MoE

| Model | Weighted Brier | R Brier | F Brier |
|---|---:|---:|---:|
| strong CatBoost baseline | **0.24789305** | **0.24798863** | **0.24716205** |
| Pitch Arsenal MoE | 0.24834853 | 0.24807175 | **0.25036706** |

**Why rejected as standalone:** nearly competitive on R, but severely worse on F. The model also overfit quickly, typically around epoch 2-4.

**Retained concept:** physical features may still be useful as a small complementary state, especially on R, but not as an end-to-end replacement.

## 12.3 Physical multitask variants

A rich physical multitask candidate scored approximately **877.1** as a standalone internal model, but received **ensemble weight 0** in the safe optimization.

**Decision:** rejected from the final ensemble; physical state had insufficient independent predictive value under the tested construction.

---

# 13. Multitask / decomposed-target path that led to the SAFE SOTA

This family was the most productive rule-safe direction after the simple dual-track model.

The central idea was to stop treating `control_success` as one monolithic target and use related historical outcome states (`reverse`, `middle`, `ball`, `strike`) plus frozen historical profiles.

## 13.1 Intermediate progression

| Step | Internal score | Brier | Interpretation |
|---|---:|---:|---|
| multi-anchor logical combination | **884.1** | — | useful but incomplete |
| anchor-cross extension | **893.8** | — | incremental gain |
| anchor-current multitask | **886.4** | **0.247592678** | no decisive gain alone |
| offset residual + success hurdle | **911.7** | **0.247529459** | complementary decomposition |
| auxiliary contextual profile | **888.0 -> 902.6** | — | profile signal useful |
| domain calibration | **956.0** | **0.247418642** | large safe gain |
| pressure auxiliary profile | **957.5** | **0.247414884** | ~34% pressure-component weight |
| final safe ensemble | **981.5** | **0.247355098** | current internal safe SOTA |

## 13.2 Hurdle and direct logic controls

- rich success-hurdle standalone: approximately **899.6**
- direct gate-logic model: approximately **952.2**

The direct gate was largely redundant once the stronger ensemble components were present.

## 13.3 Final safe model components

The verified safe submission contains the following model families/artifacts:

- `multi.cbm`
- `aux_reverse.cbm`
- `aux_middle.cbm`
- `hurdle_gate.cbm`
- `hurdle_cond.cbm`
- `offset.cbm`
- `joint.cbm`
- frozen `history.csv.gz`
- frozen `history_aux.csv.gz`
- fixed metadata / R-F ensemble weights

All historical profile generation is based on training history; hidden-test rows do not update one another.

**External LB:** **1098.86143**.

---

# 14. Direct Brier / structured outcome controls

## 14.1 Direct Brier boosting

A direct squared-error/Brier-oriented booster was executed and prediction artifacts exist.

**Status:** not promoted by itself. Exact standalone metric is not preserved in the tracked text record.

## 14.2 Direct Logloss candidate

A direct Logloss candidate received **ensemble weight 0** in later optimization.

**Decision:** rejected as redundant/inferior relative to decomposed experts.

## 14.3 Structured multiclass

A safe row-independent structured multiclass experiment was prepared with rich frozen features and depth 7/9 variants on GPU.

At the latest recorded checkpoint, final metrics were **not yet available**.

**Status:** PENDING, not part of the SOTA.

---

# 15. INVALID prefix-state branch — numerically strong but prohibited

This section is intentionally preserved because the numbers were impressive and could otherwise be rediscovered accidentally. **None of these scores are valid submission evidence.**

The implementation reconstructed state by using ordering/shift/rolling relationships among validation/test rows. That violates the rule that each hidden-test row must be independently predictable.

## 15.1 Numerical progression before invalidation

| Experiment | Standalone / combined score | Brier | Status |
|---|---:|---:|---|
| A3 multi-prefix state | **1063.6** | **~0.247150** | INVALID-RULE |
| A3 + safe ensemble | **1134.5** | **0.246973** | INVALID-RULE |
| 800-tree depth-6 prefix | standalone **1066.8**, ensemble **1139.0** | — | INVALID-RULE |
| multi-scale prefix | standalone **1097.7**, ensemble **1154.4** | **0.246923** | INVALID-RULE |
| depth tuning | **1155.3** | — | INVALID-RULE |
| RMSE/Brier prefix model | standalone **1102.3**, ensemble **1155.5** | — | INVALID-RULE |
| prefix joint expert | standalone **895**, ensemble **1167.5** | **0.246890** | INVALID-RULE |
| joint-depth diversity | **1168.4** | — | INVALID-RULE |
| global simplex reoptimization | **1186.8** | **0.246842** | INVALID-RULE |
| 2024 affine diagnostic | **1226.9** | **0.246742** | INVALID-RULE |
| conservative fixed R×1.15 / F×1.25 | **1222.8806** | **0.246752060** | INVALID-RULE |

The model also showed strong apparent fold behavior in 2022/2023, which initially made it look legitimate. The rule audit overrode performance: batch predictions depended on peer rows.

**Final action:** prefix ZIPs moved to `dist/quarantine_rule_violation/`; internal 1222.9 record declared invalid and excluded from SOTA comparison.

---

# 16. Post-invalidation safe restart

After removing the prefix family:

- safe internal baseline: approximately **981.4-981.5**
- safe Brier: **0.247355098**
- external LB: **1098.86143**

Two immediate safe follow-ups failed:

| Experiment | Result | Decision |
|---|---|---|
| season delta-count / EB basis | ensemble weight **0** | rejected |
| frozen high-order context lattice | standalone score about **890** | rejected |

These results reinforced the value of the existing decomposed frozen-profile ensemble rather than additional high-order lookup complexity.

---

# 17. Runs/branches discovered but exact metrics not preserved in Git

The local tree proves that the following experiments or variant families were executed and produced artifacts. Their numeric CSV/NPZ outputs were intentionally excluded from Git, and the available conversation/handoff text does not contain enough information to quote a trustworthy exact final metric for every variant.

They should be treated as **historical executed attempts, not unknown new ideas**:

### Regime / temporal

- `abs_regime_shift`
- `game_type_domain_suite`
- `game_type_experience`
- `game_type_temporal_regime_ablation`
- `regime_context_cross_suite`
- `regime_feature_prediction_suite`
- `rf_conditional_frozen_stable_gate`
- `rf_level_evidence`
- `old_f_data_selection_probe`
- `row_id_time_f2024`
- `rowlocal_asof_count_probe`
- `trajectory_experience_gate`
- `pitcher_trajectory_probe`

### R-experience

- `r_experience_band_ablation`
- `r_experience_eb_shrinkage`
- `r_experience_usage_audit`
- `r_only_experience_specialists`

### Frozen historical state

- `frozen_season_anchor_probe`
- `frozen_domain_path_probe`
- `frozen_domain_path_factorial`
- `frozen_domain_path_rate_probe`
- `frozen_domain_team_path_probe`
- `previous_season_cheatsheet_probe`
- `relative_decayed_cheatsheet_probe`
- `profile_direct`

### Decomposition variants

- many `multitask_outcome_boosting_*` repetitions over depth, repeats, seeds, anchors, cross features, matchup/count/pressure/domain profiles, auxiliary-profile variants, arsenal and leverage variants
- many `offset_residual_*` variants
- many `success_hurdle_*` variants
- `aux_heads_reverse-middle`
- `aux_heads_ball-strike`
- F2023 auxiliary-head variants
- `joint_success_*`
- `joint_outcome_*`
- `hierarchical_residual_transfer`
- `backbone_shrinkage_probe`

### Physical / Trackman

- `physical_multitask_current`
- `physical_multitask_prev`
- `physical_multitask_current_rich`
- `physical_state_distillation`
- `physical_regime_suite`

### Other model families

- `xgb_regime_probe`
- `wide_deep_brier`
- `qwen3_1p7b_rag`
- `qwen3_1p7b_rag_ultrafast`

**Repository-cleanup rule:** do not keep all of these as active top-level scripts. Preserve this inventory and the most informative result tables, then archive/reduce obsolete runners after the safe SOTA has been reproduced from a clean clone.

---

# 18. Final conclusions from the full experiment history

1. **Raw tabular + CatBoost remains the strongest backbone.** VAE/latent neural heads, LLM inference, generic sequence models, and standalone physical neural models did not beat it reliably.
2. **2023 is a genuine structural breakpoint.** In particular, F changes dramatically; however, raw `game_type` still helps in the newest regime and should not simply be dropped.
3. **Use all history, but add recent/regime specialists conservatively.** Hard history deletion and smooth recency decay both lost useful signal.
4. **Hard player-experience routing is harmful.** Experience should remain a continuous/shrunk context feature, not a separate expert boundary.
5. **The largest rule-safe improvement came from target decomposition and frozen historical profiles:** multitask auxiliary outcomes, hurdle, residual offset, joint outcome, matchup/count/pressure/domain profiles, and R/F-specific ensemble weights.
6. **Trackman linkage is technically successful but predictive value is not yet proven.** Physical models received zero or low ensemble weight under current tests.
7. **Prefix-state results must never be cited as valid SOTA.** The 1222.9 result is numerically real for that validation code but competition-invalid because test rows influence one another.
8. **Current authoritative baseline:** safe Brier **0.247355098**, internal score **981.5**, external LB **1098.86143**.
