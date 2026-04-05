# LivePortrait Deep Dive — Direct Integration Analysis

**Date:** 2026-04-05
**Status:** Architecture research
**Related:** [Template Rig Architecture](2026-04-05-template-rig-architecture.md), [ComfyUI Workflows for Training Data](2026-04-05-comfyui-workflows-for-training-data.md)

---

## Summary

LivePortrait is not a diffusion model. It is a deterministic implicit-keypoint warp+decode pipeline (~500MB weights, 12.8ms/frame on RTX 4090) that exposes facial motion as a small set of tensors we can directly manipulate. The PowerHouseMan ComfyUI ExpressionEditor is ~300 lines of Python over LivePortrait's pipeline — parametric sliders are reverse-engineered hand-tuned offsets applied to the expression deformation tensor.

**Recommendation**: vendor LivePortrait directly into the project and port the slider→δ mapping, skipping ComfyUI entirely for batch training-data generation. ComfyUI + ExpressionEditor is retained only for *interactive verb authoring* (finding slider values visually), then those values are copied to `verbs.toml` and replayed from Python.

---

## Architecture (from arXiv:2407.03168)

LivePortrait's pipeline has four neural modules:

| Module | Symbol | Role |
|---|---|---|
| Appearance Extractor | ℱ | Source image → appearance feature volume |
| Motion Extractor | ℳ | Image → {keypoints, rotation, expression δ, translation, scale, eye-open, lip-open} |
| Warping Module | 𝒲 | Warps appearance features by motion transform |
| Decoder | 𝒢 | Warped features → RGB output |

### Motion Representation

For a source image, ℳ(s) outputs:

```
xc,s  ∈ ℝ^(K×3)   canonical implicit keypoints
Rs    ∈ ℝ^(3×3)   head rotation matrix
δs    ∈ ℝ^(K×3)   per-keypoint expression deltas
ts    ∈ ℝ^3       translation
ss    ∈ ℝ         scale
cs,eyes, cs,lip   eye/lip opening scalars
```

Transform equation: `xs = ss · (xc,s · Rs + δs) + ts`

The expression tensor δs (K≈21 for the editable subset) is the thing PHM's ExpressionEditor manipulates directly.

### Retargeting MLPs

LivePortrait ships with three tiny MLPs (all <1MB) that operate on keypoints:

| Module | Architecture | Input | Output |
|---|---|---|---|
| Stitching | [126, 128, 128, 64, 65] | xs ⊕ xd concat | Δst ∈ ℝ^(K×3) |
| Eyes retargeting | [66, 256, 256, 128, 128, 64, 63] | xs + eye scalars | Δeyes |
| Lip retargeting | [65, 128, 128, 64, 63] | xs + lip scalar | Δlip |

The eye/lip retargeting modules take a scalar "how open" and produce keypoint deltas. **These are LivePortrait's native parametric controls** — the simplest path to eye/mouth verbs.

### Training data and performance

- 69M mixed image-video frames
- 12.8ms inference per frame on RTX 4090 (PyTorch)
- Fine-tuned on animals (cats, dogs, pandas) — supports stylized inputs
- Works on anime/stylized sources in community tests

---

## PHM ExpressionEditor Mechanism

The ExpressionEditor node in `ComfyUI-AdvancedLivePortrait/nodes.py` implements parametric control by applying hand-tuned additive offsets to specific keypoint indices of δ:

```python
# Pseudocode reconstructed from PHM's calc_fe()
def calc_fe(exp_tensor, sliders):
    # exp_tensor: (21, 3) — expression deformation from motion extractor
    # Each slider maps to specific keypoint indices with precise coefficients:

    # smile → indices [20, 14, 17, 13, 16, 3, 7] with coefficients like ±0.01
    exp_tensor[20, 1] += smile * 0.01
    exp_tensor[14, 1] += smile * 0.02
    # ... ≈15-20 targeted modifications per slider

    # blink → eyelid keypoints
    # eyebrow → brow keypoints
    # aaa/eee/woo → mouth shape keypoints
    # pupil_x/y → iris keypoints
    return exp_tensor

def edit_expression(source_img, sliders):
    kp_info = pipeline.get_kp_info(source_img)        # {kp, R, exp, t, scale}
    exp_mod = calc_fe(kp_info['exp'], sliders)
    R_mod = rotation_matrix(pitch, yaw, roll)          # combines with kp_info['R']
    return pipeline.warp_decode(kp_info, exp_mod, R_mod)
```

### Slider inventory (from PHM's node registration)

| Slider | Range | Default | Semantics |
|---|---|---|---|
| `rotate_pitch` | -20 to 20 | 0 | head up/down |
| `rotate_yaw` | -20 to 20 | 0 | head left/right |
| `rotate_roll` | -20 to 20 | 0 | head tilt |
| `blink` | -20 to 5 | 0 | eye closure (negative = more closed) |
| `eyebrow` | -10 to 15 | 0 | brow elevation |
| `wink` | 0 to 25 | 0 | single-eye closure |
| `pupil_x` | -15 to 15 | 0 | gaze horizontal |
| `pupil_y` | -15 to 15 | 0 | gaze vertical |
| `aaa` | 0 to 120 | 0 | mouth open (vowel "ah") |
| `eee` | -20 to 15 | 0 | mouth wide (vowel "ee") |
| `woo` | -20 to 15 | 0 | mouth pucker (vowel "oo") |
| `smile` | -0.3 to 1.3 | 0 | smile intensity |
| `src_ratio` | 0 to 1 | 1 | source expression retention |
| `sample_ratio` | -0.2 to 1.2 | 1 | sample image blend weight |
| `crop_factor` | 1.5 to 2.5 | 1.7 | face crop expansion |

These are MIT-licensed and portable to direct Python.

---

## Integration Plan (Direct Python, no ComfyUI runtime)

### Directory structure

```
portrait-to-live2d/
├── third_party/
│   └── LivePortrait/          # git submodule @ pinned commit
├── mlp/data/
│   └── live_portrait/
│       ├── __init__.py
│       ├── pipeline.py        # thin wrapper around LivePortrait.Pipeline
│       ├── verb_sliders.py    # port of PHM calc_fe()
│       └── verb_renderer.py   # verb dict → rendered image batch
└── templates/humanoid-anime/
    └── verbs.toml             # verb → {slider values} + {template param targets}
```

### Core API

```python
# mlp/data/live_portrait/verb_renderer.py
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch

@dataclass
class VerbSliders:
    rotate_pitch: float = 0.0
    rotate_yaw: float = 0.0
    rotate_roll: float = 0.0
    blink: float = 0.0
    eyebrow: float = 0.0
    wink: float = 0.0
    pupil_x: float = 0.0
    pupil_y: float = 0.0
    aaa: float = 0.0
    eee: float = 0.0
    woo: float = 0.0
    smile: float = 0.0

@dataclass
class SourceState:
    """Precomputed source image kp_info — reusable across many verb renders."""
    kp: torch.Tensor            # (1, 21, 3) canonical keypoints
    R: torch.Tensor             # (1, 3, 3) rotation
    exp: torch.Tensor           # (1, 21, 3) expression delta
    t: torch.Tensor             # (1, 3) translation
    scale: torch.Tensor         # (1,) scale
    appearance_feat: torch.Tensor  # (1, C, D, H, W) appearance features
    crop_info: dict             # face crop transform for pasting back

class VerbRenderer:
    def __init__(self, checkpoint_dir: Path, device: str = "cuda"):
        from third_party.LivePortrait.src.pipeline import LivePortraitPipeline
        self.pipeline = LivePortraitPipeline(checkpoint_dir, device)
        self.device = device

    def precompute_source(self, source_img: np.ndarray) -> SourceState:
        """Run ℳ(s) and ℱ(s) once; reuse across all verbs for this source."""
        return self.pipeline.prepare_source(source_img)

    def render(self, source: SourceState, sliders: VerbSliders) -> np.ndarray:
        exp_mod = apply_sliders(source.exp, sliders)
        R_mod = source.R @ rotation_matrix(
            sliders.rotate_pitch, sliders.rotate_yaw, sliders.rotate_roll
        )
        x_d = source.scale * (source.kp @ R_mod + exp_mod) + source.t
        img = self.pipeline.warp_decode(source.appearance_feat, source.kp, x_d)
        return self.pipeline.paste_back(img, source.crop_info)
```

### Verb library format

```toml
# templates/humanoid-anime/verbs.toml

[verbs.neutral]
# all sliders default 0

[verbs.close_eyes_both]
sliders = { blink = -15.0 }
targets = { EyeLOpenLeft = 0.0, EyeLOpenRight = 0.0 }

[verbs.wink_left]
sliders = { wink = 20.0 }
targets = { EyeLOpenLeft = 0.0, EyeLOpenRight = 1.0 }

[verbs.smile_slight]
sliders = { smile = 0.5, eee = 8.0 }
targets = { MouthFormSmile = 0.3, EyeLSmile = 0.2, EyeRSmile = 0.2 }

[verbs.smile_wide]
sliders = { smile = 1.2, aaa = 40.0 }
targets = { MouthFormSmile = 1.0, MouthOpenY = 0.3 }

[verbs.surprised]
sliders = { blink = 3.0, aaa = 60.0, eyebrow = 12.0 }
targets = { EyeLOpenLeft = 1.0, EyeLOpenRight = 1.0, MouthOpenY = 0.7, BrowLY = 1.0 }

[verbs.look_left]
sliders = { pupil_x = -12.0 }
targets = { EyeBallX = -0.8 }
```

### Batch generation pipeline

```python
# mlp/data/generate_verb_samples.py
def generate_verb_dataset(
    reference_img: Path,
    verbs_toml: Path,
    n_variations: int,
    output_dir: Path,
) -> None:
    renderer = VerbRenderer(LP_WEIGHTS_DIR)
    source = renderer.precompute_source(load_image(reference_img))
    verbs = load_verbs(verbs_toml)

    for verb_name, (sliders, targets) in verbs.items():
        for i in range(n_variations):
            # Jitter sliders ±10% for variation
            jittered = jitter_sliders(sliders, std=0.10)
            img = renderer.render(source, jittered)

            # Extract 1014-d MediaPipe signal
            lm, bs, pose = mediapipe_extract(img)
            if lm is None:
                continue  # face detection failed — skip
            x = np.concatenate([lm, bs, pose])  # (1014,)

            # Quality filter — MediaPipe must confirm the verb happened
            if not validates_verb(bs, verb_name):
                continue

            save_sample(output_dir, x, targets, verb=verb_name, idx=i)
```

### Interactive verb authoring workflow

ComfyUI + ExpressionEditor is still useful for one task: *discovering* the right slider values for a new verb.

1. Load reference image in ComfyUI with ExpressionEditor node
2. Tweak sliders live until the output matches the target verb
3. Copy slider values into `verbs.toml`
4. Batch generation runs from Python without ComfyUI

This is a one-time authoring cost per verb, not per training sample.

---

## Dependencies and Licensing

### Added dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    # existing...
    "torch>=2.1",
    "torchvision>=0.16",
    "insightface>=0.7.3",      # face detection/cropping
    "onnxruntime-gpu>=1.17",   # insightface inference
    "opencv-python>=4.9",
]
```

### Model weights

- LivePortrait weights: ~500MB (appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module)
- Source: `huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights/`
- InsightFace buffalo_l face detector: ~300MB

### License verification needed

- **LivePortrait**: MIT License (as of Kwai VGI repo, needs verification on vendor commit)
- **PHM ExpressionEditor**: MIT License — port of `calc_fe()` is permitted with attribution
- **InsightFace buffalo_l**: research-only license on some models — **we need to verify before commercial release**
- **Alternative**: MediaPipe face detection (already a project dependency) can replace InsightFace if license is a blocker

### VRAM requirements

- LivePortrait inference: ~4GB VRAM at 512² output resolution
- Batch size 8: ~10GB VRAM
- Runs on consumer GPUs (3070+ recommended, 4070+ comfortable)

---

## Throughput Estimates

For a 10,000-sample training dataset from one reference image:

| Step | Time per sample | Total |
|---|---|---|
| `precompute_source` | 50ms (once) | 50ms |
| `render(sliders)` | ~15ms (batched to 8: ~5ms amortized) | 50s (batched) |
| MediaPipe extract | ~8ms | 80s |
| Quality filter + save | ~2ms | 20s |
| **Total** | | **~2.5 min** |

For 60 verbs × 167 variations = 10k samples: one RTX 4090 completes in ~3 minutes. This makes per-rig customization feasible (user provides a reference photo, we generate a personalized dataset in minutes).

---

## Open Questions

1. **Does LivePortrait's motion extractor work on Live2D anime renders?** If yes, we can skip human reference photos entirely and use the Hiyori render at neutral as source. Needs empirical test.
2. **Slider coverage for template params** — do PHM's 12 sliders cover the 74 Hiyori params, or do we need compound verbs + pose renders to fill gaps? (Head pose params are already covered by existing rig-render pipeline.)
3. **InsightFace license** — check if `buffalo_l` bundled model blocks our distribution. Fallback: MediaPipe face detection produces compatible crops.
4. **Stylized source robustness** — anime renders have different lighting/shading than LivePortrait's training distribution. Fallback: style-transfer the render to photo first, then apply verbs.
5. **Keypoint index stability** — PHM's hardcoded indices (20, 14, 17, ...) correspond to LivePortrait's motion extractor output order. If we upgrade LivePortrait weights later, these may need re-validation.

---

## Next Steps

1. Vendor LivePortrait as `third_party/LivePortrait` git submodule
2. Verify licenses (LivePortrait + InsightFace)
3. Port `calc_fe()` to `mlp/data/live_portrait/verb_sliders.py` (~200 LOC)
4. Write `VerbRenderer` wrapper (~150 LOC)
5. Author initial 30-verb library for `humanoid-anime` template
6. Run first-experiment: 10 verbs × 1 reference, validate MediaPipe blendshape response ≥8/10
7. Scale to 60 verbs × 167 variations = 10k sample dataset
8. Retrain MLP on verb-generated + pose-rendered merged dataset

---

## Sources

[1] Guo et al. "LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control." arXiv:2407.03168, 2024.
[2] KwaiVGI/LivePortrait repository. https://github.com/KwaiVGI/LivePortrait
[3] PowerHouseMan/ComfyUI-AdvancedLivePortrait. https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait
[4] LivePortrait project page. https://liveportrait.github.io/
