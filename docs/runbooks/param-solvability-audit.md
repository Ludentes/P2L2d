# Runbook: Template Parameter Solvability Audit

When onboarding a new rig template, run this process to determine which
Live2D parameters can be reliably predicted by the MLP and which need
alternative control sources.

## Why

Not every Live2D parameter can be driven by face tracking. Some (like brow
tilt) have no reliable signal in MediaPipe output. Training the MLP on
unsolvable params adds noise and wastes model capacity. Better to identify
them early, drop from the MLP, and wire them to alternative sources.

## Prerequisites

- A trained MLP or at minimum a 10k-sample NPZ dataset
- Template `schema.toml` and `verbs.toml` for the rig
- Python environment with numpy, mediapipe

## Step 1: Generate Training Data

```bash
uv run python -m mlp.data.live_portrait.generate_verb_samples \
    --reference assets/generated-faces \
    --n 10000 \
    --out mlp/data/live_portrait/datasets/audit_10k.npz \
    --bs-only  # 58-d blendshape+pose features (faster, sufficient for audit)
```

## Step 2: Compute Blendshape–Param Correlations

For each template param, compute Pearson correlation with all 52 MediaPipe
blendshapes. This reveals whether the face tracker can even *see* the param.

```python
import numpy as np
d = np.load("mlp/data/live_portrait/datasets/audit_10k.npz")
features = d["features"]  # (N, 58) in bs-only mode
labels = d["labels"]
param_names = d["param_names"].tolist()

# Blendshapes are the first 52 dims
bs = features[:, :52]

for j, pname in enumerate(param_names):
    corrs = [(abs(np.corrcoef(bs[:, k], labels[:, j])[0,1]), k)
             for k in range(52)]
    corrs.sort(reverse=True)
    max_r = corrs[0][0]
    print(f"{pname:20s}  max|r|={max_r:.3f}  {'OK' if max_r > 0.3 else 'WEAK' if max_r > 0.15 else 'UNSOLVABLE'}")
```

## Step 3: Train and Evaluate

```bash
uv run python -m mlp.train_verb_mlp \
    --data mlp/data/live_portrait/datasets/audit_10k.npz \
    --out mlp/checkpoints/audit \
    --epochs 200
```

Review per-param R² in `mlp/checkpoints/audit/metrics.csv`.

## Step 4: Classify Parameters

| R² Range | Max |r| | Classification | Action |
|---|---|---|---|
| ≥ 0.5 | ≥ 0.4 | **Solvable** | Keep in MLP |
| 0.2–0.5 | 0.2–0.4 | **Marginal** | Keep, but expect noise. Consider loss weighting. |
| < 0.2 | < 0.2 | **Unsolvable** | Drop from MLP schema |

## Step 5: Wire Dropped Params to Alternative Sources

Unsolvable params still need to be driven at runtime. Options:

1. **BCI signals** — Muse VTuber Bridge sends custom VTS params
   (`MuseBlink`, `MuseClench`, `MuseFocus`, `MuseRelaxation`)
2. **Dedicated detectors** — e.g. a gaze-tracking model for EyeBallY,
   or IMU data for fine head angles
3. **Rule-based coupling** — derive from other params
   (e.g. `EyeBallY ≈ -0.03 * AngleY` to track head pitch)
4. **Manual sliders** — user sets fixed value in VTuber app (VTS custom params)
5. **Default value** — param stays at rig default (0.0 for most)

Update `schema.toml` to document dropped params and their intended control
source. The MLP should only predict what it can reliably predict.

## Reference: humanoid-anime Template Audit (2026-04-05)

Results from 10k samples, 100 ComfyUI-generated faces, 54 verbs:

| Param | Max |r| | R² (58-d) | R² (1014-d) | Classification |
|---|---|---|---|---|
| AngleX/Y/Z | — | 0.99+ | 0.99+ | Solvable (pose matrix) |
| MouthOpenY | 0.83 | 0.76 | 0.83 | Solvable |
| MouthForm | 0.76 | 0.70 | 0.72 | Solvable |
| Cheek | 0.73 | 0.77 | 0.79 | Solvable |
| EyeROpen | 0.73 | 0.52 | 0.61 | Solvable |
| EyeLSmile | 0.54 | 0.58 | 0.57 | Solvable |
| EyeRSmile | 0.54 | 0.58 | 0.58 | Solvable |
| EyeLOpen | 0.59 | 0.40 | 0.55 | Marginal |
| EyeBallX | 0.42 | 0.34 | 0.40 | Marginal |
| BrowLY | 0.34 | 0.24 | 0.27 | Marginal |
| BrowRY | 0.34 | 0.24 | 0.26 | Marginal |
| **EyeBallY** | **0.18** | **0.13** | **0.13** | **Dropped** → rule/gaze detector |
| **BrowLAngle** | **0.12** | **0.04** | **0.04** | **Dropped** → BCI/manual |
| **BrowRAngle** | **0.12** | **0.04** | **0.04** | **Dropped** → BCI/manual |
