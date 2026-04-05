# Full Training & Runtime Pipeline Plan

**Date:** 2026-04-05
**Status:** Plan, pre-implementation
**Supersedes:** ComfyUI-centric data-generation approaches
**Related:** [LivePortrait Deep Dive](2026-04-05-liveportrait-deep-dive.md), [Template Rig Architecture](2026-04-05-template-rig-architecture.md), [Anime Landmark Failure](2026-04-05-anime-landmark-eye-tracking-failure.md)

---

## Executive Summary

Training-data generation uses **LivePortrait** (direct Python, no ComfyUI runtime) to produce expression samples from a human reference photo, plus **Live2D rig renders** for head-pose samples. The merged dataset trains a template MLP (1014-d MediaPipe input → template standard params). User rigs map to templates via a manifest file.

**Style transfer is not in the training-data hot path.** It's retained only as an optional offline tool for reference-image acquisition, and the research in `2026-04-05-comfyui-workflows-for-training-data.md` serves as a fallback plan if LivePortrait proves inadequate.

---

## Pipeline Overview

```
╔══════════════════════════════════════════════════════════════════════════╗
║ OFFLINE: Template Authoring (once per archetype)                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Author verb library (interactive, via ComfyUI ExpressionEditor)      ║
║       templates/{archetype}/verbs.toml                                   ║
║       ≈30-60 verbs × {sliders: {...}, targets: {...}}                    ║
║                                                                          ║
║  2. Acquire reference image (human photo; one-time style-transfer        ║
║       only if matching rig identity matters to the archetype)            ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║ TRAINING DATA GENERATION (per template, ~3 min on RTX 4090)              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  reference.jpg ──► LivePortrait.precompute_source() ──► SourceState      ║
║                                                               │          ║
║  For each verb × N variations:                                │          ║
║    jitter(sliders) ──► VerbRenderer.render(source, sliders) ──┘          ║
║                                  │                                       ║
║                                  ▼                                       ║
║                            rendered image (512²)                         ║
║                                  │                                       ║
║                                  ▼                                       ║
║                      MediaPipe FaceLandmarker                            ║
║                  → landmarks (956) + blendshapes (52) + pose (6)         ║
║                  = 1014-d input vector                                   ║
║                                  │                                       ║
║                                  ▼                                       ║
║                      quality filter (blendshapes confirm verb)           ║
║                                  │                                       ║
║                                  ▼                                       ║
║                     labeled sample: (x_1014, y_template_params)          ║
║                                                                          ║
║  For head-pose verbs only (AngleX/Y/Z, BodyAngle*, Position*):           ║
║    Live2D rig render with known pose params ──► MediaPipe ──► sample     ║
║                                                                          ║
║  Merge: verb-generated expression samples + rig-rendered pose samples    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║ MLP TRAINING (per template, once)                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  4-layer MLP: 1014 → 512 → 256 → N_template_params                       ║
║  Train on merged dataset → templates/{archetype}/model.pt                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║ RUNTIME (real webcam → VTS, ~16ms/frame)                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  webcam frame                                                            ║
║       │                                                                  ║
║       ▼                                                                  ║
║  MediaPipe FaceLandmarker → 1014-d                                       ║
║       │                                                                  ║
║       ▼                                                                  ║
║  Load user's rig.manifest.toml → identifies template                     ║
║       │                                                                  ║
║       ▼                                                                  ║
║  templates/{archetype}/model.pt (MLP)                                    ║
║       │                                                                  ║
║       ▼                                                                  ║
║  template standard params (N_template values)                            ║
║       │                                                                  ║
║       ▼                                                                  ║
║  manifest.param_map → user's custom Live2D param names                   ║
║       │                                                                  ║
║       ▼                                                                  ║
║  templates/{archetype}/curves.toml → response curves                     ║
║       │                                                                  ║
║       ▼                                                                  ║
║  VTS WebSocket API (Muse bridge handles this)                            ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Role of Style Transfer (what changed)

### Previous assumption (now discarded)

"Live2D render anime → MediaPipe fails → style-transfer to photo → MediaPipe works" — putting diffusion in the critical path of every training sample.

### Current role

Style transfer appears in the pipeline in two narrow places, both optional:

1. **Reference acquisition (one-time, offline)** — if the user wants the training distribution to match their anime character's face shape specifically, style-transfer Hiyori's neutral render → photorealistic portrait *once*, and use that as LivePortrait's source. Cost: one diffusion call per user rig, not per sample.

2. **Held-out adversarial validation** — to quantify the "LivePortrait-warped photo → real webcam" domain gap, we keep a separate test set of real anime renders. Style-transfer is not needed here; raw anime renders serve as adversarial inputs to measure degradation.

If Open Question #1 ("Does LivePortrait's motion extractor work on Live2D anime renders?") resolves positively, role #1 disappears and style transfer leaves the pipeline entirely.

The ComfyUI style-transfer research in `2026-04-05-comfyui-workflows-for-training-data.md` remains a **fallback plan**: if LivePortrait proves inadequate (e.g., identity drift across verbs, failure on stylized sources, MediaPipe can't read warped output), we fall back to FLUX Kontext verb-prompting.

---

## Architectural Consequences

### No ComfyUI at runtime or in CI

ComfyUI becomes a human-facing verb-authoring tool (desktop app): the rigger tweaks ExpressionEditor sliders live, copies the values into `verbs.toml`. Batch data generation is pure PyTorch. CI/CD does not need a ComfyUI server.

### Per-rig dataset generation is cheap

~3 minutes on RTX 4090 per 10k samples. Each user can have a training set matched to their character's reference photo. If template-level generalization is insufficient, per-rig MLPs become feasible.

### Domain gap relocated, not eliminated

Previously the gap was "anime → photo". Now it is "LivePortrait-warped photo → real webcam". Mitigations differ:

- Lighting/color augmentation during training
- Multi-reference training (N reference photos per template, not one)
- Test-time webcam calibration (per-session linear correction) — future work

### Validation strategy

Two held-out test sets:

1. **In-distribution (strict)** — verb renders not seen during training, same reference image. Measures MLP generalization within the LivePortrait domain.
2. **Adversarial (real-world proxy)** — actual Live2D anime renders at known params. Measures domain transfer to the deployment target. Low R² here is expected; the metric is improvement over baseline rig-render training.

---

## Implementation Phases

### Phase 1 — LivePortrait integration (smallest testable unit)

- Vendor LivePortrait as `third_party/LivePortrait` git submodule
- Verify LivePortrait + InsightFace licenses
- Port PHM's `calc_fe()` to `mlp/data/live_portrait/verb_sliders.py`
- Write `VerbRenderer` wrapper class (~150 LOC)
- **Smoke test**: 10 verbs × 1 reference image → verify MediaPipe blendshape response ≥8/10 matches expected verb

### Phase 2 — Verb library authoring for humanoid-anime

- Install ComfyUI + PHM AdvancedLivePortrait (authoring tool only)
- Author 30-verb base library + ~15 compound verbs
- Define verb → template param target mappings
- Commit `templates/humanoid-anime/verbs.toml`

### Phase 3 — Dataset generation pipeline

- Write `mlp/data/generate_verb_samples.py`
- Integrate MediaPipe blendshape-based quality filter
- Generate 10k-sample dataset from Hiyori-style reference
- Check data quality via existing `mlp/data/check_data_quality.py`

### Phase 4 — MLP retraining on 1014-d input

- Extend MLP input layer 956 → 1014 (add 52 blendshapes + 6 pose)
- Retrain on merged verb-generated + rig-pose dataset
- Compare against current landmarks-only baseline

### Phase 5 — Template + manifest system

- Formalize `templates/humanoid-anime/` directory with `schema.toml`, `verbs.toml`, `model.pt`, `curves.toml`
- Write `rig/manifest.py` to load user manifest and remap template params → user custom param names
- Validate with a non-Hiyori rig (second humanoid character)

### Phase 6 — Runtime pipeline wiring

- Extract 1014-d signal from webcam (MediaPipe landmarker with blendshapes enabled)
- Load template MLP + curves based on user manifest
- Wire into existing VTS WebSocket output
- End-to-end latency target: <30ms per frame on CPU-only inference

### Phase 7 — Additional templates (future)

- humanoid-realistic, kemonomimi (ears, tail)
- Evaluate anthro-furry viability (MediaPipe on non-human faces)

---

## Open Questions (tracked for empirical resolution)

1. **Does LivePortrait work on anime renders directly?** — If yes, style transfer drops fully.
2. **Slider coverage vs template param count** — do 12 PHM sliders cover 20–30 expression params, or do we need compound verbs?
3. **Identity robustness across 10k verb applications** — any drift as sliders get jittered?
4. **Single reference vs multi-reference per template** — single may overfit to reference face geometry.
5. **MediaPipe reliability gradient** on LivePortrait output — empirical blendshape response test.
6. **InsightFace license** — whether `buffalo_l` blocks distribution (fallback: MediaPipe face detection).

---

## Success Metrics

A template is "working" when:

- 60-verb library trained MLP achieves R² ≥ 0.5 on held-out in-distribution verb samples across all face-tracked params (was: expected ≈0 for eye/mouth on landmarks-only baseline)
- Adversarial anime-render test set achieves R² ≥ 0.3 (measures domain transfer)
- Real webcam demo on reference rig produces visually plausible animation (subjective but necessary)
- A second rig with only a manifest (no retraining) animates acceptably

---

## Dependencies to Add

```toml
# pyproject.toml
dependencies = [
    # existing: torch, mediapipe, live2d-py, ...
    "insightface>=0.7.3",       # face detection for LivePortrait cropper
    "onnxruntime-gpu>=1.17",    # insightface inference backend
]
```

Plus LivePortrait model weights (~500MB) via `huggingface-cli download KlingTeam/LivePortrait`.

No ComfyUI, no SDXL/FLUX weights needed for the core pipeline.
