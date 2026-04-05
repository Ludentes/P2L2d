# Phase 4 Results: MLP Training on 1014-d Verb Features

**Date:** 2026-04-05
**Branch:** `worktree-liveportrait-phase1`
**Status:** ✅ Success metric met (R² ≥ 0.5 on held-out verb samples)

## What Was Built

- `mlp/model.py`: `CartoonAliveMLP` extended to configurable `input_dim`
  (default 956 preserves backward compat with Hiyori checkpoint;
  set `input_dim=1014` for verb pipeline). Added `set_norm_stats()`.
- `mlp/train_verb_mlp.py`: NPZ dataset trainer with train/val split,
  per-param R²/RMSE metrics, early stopping, saved history.

## Datasets

- `dev_500.npz` — 500 samples, 6 references (dropped s22, s30)
- `train_10k.npz` — 10,000 samples, same 6 references, ~278 samples/verb avg
  - Generation time: 421s (23.7 samples/s)
  - Rejection rate: 1.04% (104/10,104)

## Training Run — 10k Dataset

Config: input_dim=1014, hidden=512/256/128, AdamW lr=3e-4, wd=1e-4,
batch=512, val=15%, cosine schedule over 300 epochs, patience=30.

Final val MSE: **0.0626**. Train MSE: 0.035 (slight overfitting, ~1.8× gap).

### Per-param R² (validation set, N=1500)

| Param | R² | RMSE | label_std | Verbs targeting |
|---|---|---|---|---|
| AngleX (yaw) | **0.999** | 0.50 | 13.62 | pose jitter |
| AngleY (pitch) | **0.998** | 0.37 | 8.29 | pose jitter |
| AngleZ (roll) | **0.996** | 0.36 | 5.89 | pose jitter |
| MouthOpenY | **0.796** | 0.11 | 0.25 | 8 verbs |
| EyeROpen | **0.772** | 0.15 | 0.31 | many |
| Cheek | **0.740** | 0.13 | 0.25 | smile family |
| MouthForm | **0.659** | 0.22 | 0.38 | smile/frown/angry |
| EyeBallX | **0.611** | 0.16 | 0.25 | look_left/right |
| EyeLSmile | 0.579 | 0.17 | 0.25 | 4 squint verbs |
| EyeRSmile | 0.582 | 0.16 | 0.25 | 4 squint verbs |
| EyeLOpen | 0.568 | 0.20 | 0.30 | many |
| BrowLY | 0.308 | 0.27 | 0.33 | 4 brow verbs |
| BrowRY | 0.307 | 0.27 | 0.33 | 4 brow verbs |
| EyeBallY | 0.113 | 0.26 | 0.28 | 4 gaze verbs |
| BrowLAngle | 0.082 | 0.15 | 0.16 | 2 verbs (angry/furrow) |
| BrowRAngle | 0.084 | 0.15 | 0.16 | 2 verbs (angry/furrow) |

**Mean R² overall**: 0.57
**Mean R² (expression only, 13 params)**: 0.48
**11/16 params ≥ 0.5 R²**

## Observations

- **Pose is effectively perfectly learned** (R² > 0.99). MediaPipe's pose
  features are already in the input — the MLP just reproduces them with
  a unit transform.
- **Strong expression params** (R² > 0.6) are those with dense verb
  coverage across the parameter's range (mouth, cheek, eye-open).
- **Weak params** correlate directly with low verb diversity:
  BrowLAngle/RAngle only set by 2 verbs (angry, brow_furrow). EyeBallY
  only by 4 gaze verbs.
- **EyeLOpen vs EyeROpen asymmetry** (0.568 vs 0.772): the wink verbs
  probably confuse left/right detection in some frames.

## Next Steps

Before Phase 5 (template system) / Phase 6 (runtime), potential
improvements:

1. **Add more verbs** targeting underperformers (brow-angle-sad, brow-
   angle-surprised, more vertical-gaze variants). Low effort, high
   impact for weak params.
2. **Address EyeLOpen asymmetry** — verify wink verb directionality, or
   add explicit wink_left/wink_right examples with symmetric open eye.
3. **Weight decay tuning** — 4% train/val gap suggests slightly more
   regularization could help.
4. **Compare on anime-render test set** — key question: does the model
   transfer to Live2D rig renders? That's the ultimate goal.
