# Template + Manifest System Design (Phase 5)

**Date:** 2026-04-05
**Status:** Approved for implementation
**Scope:** Formalize humanoid-anime template, add manifest-based param remapping, update inference pipeline

---

## Goal

Bundle the trained MLP, schema, and response curves into a self-contained template. Add a manifest system so any rig can map its custom param names to the template's canonical names. Update the inference pipeline to load from templates instead of hardcoded paths.

## Approach

Minimal: one template loader module, one manifest loader, updated predictor. No registry, no CLI, no multi-template abstractions until a second template exists.

## Components

### 1. `templates/humanoid-anime/curves.toml` (new file)

Materialize `HIYORI_DEFAULTS` from `mlp/curves.py` into TOML format. Already supported by `ResponseCurveSet.from_toml()`.

```toml
[params.EyeLOpen]
type = "gamma"
gamma = 0.7
lo = 0.0
hi = 1.0

# ... etc for all 8 curved params
```

Note: param names here use template names (no `Param` prefix), matching `schema.toml`.

### 2. `templates/humanoid-anime/model.pt` (copy/reference)

The trained checkpoint. Currently at `mlp/checkpoints/humanoid-anime-bs58/model.pt`. The template loader will look for `model.pt` in the template dir. Keep the checkpoints dir as the training output location; copy the best model to the template dir as a release step.

### 3. `templates/loader.py` (new module)

```python
@dataclass
class Template:
    name: str
    schema: TemplateSchema
    model_path: Path
    curves: ResponseCurveSet

def load_template(name: str) -> Template
```

- Looks in `templates/{name}/` for `schema.toml`, `model.pt`, `curves.toml`
- `curves.toml` is optional (identity curves if missing)
- `model.pt` is optional (template usable for data generation without trained model)

### 4. `rig/manifest.py` (new module)

```python
@dataclass
class Manifest:
    template_name: str
    param_map: dict[str, str]  # rig_param → template_param

    def remap(self, template_output: dict[str, float]) -> dict[str, float]:
        """template param names → rig param names"""

def load_manifest(path: Path) -> Manifest
```

Manifest TOML format:
```toml
template = "humanoid-anime"

[param_map]
ParamAngleX = "AngleX"
ParamAngleY = "AngleY"
ParamAngleZ = "AngleZ"
ParamEyeLOpen = "EyeLOpen"
ParamEyeROpen = "EyeROpen"
ParamEyeBallX = "EyeBallX"
ParamEyeLSmile = "EyeLSmile"
ParamEyeRSmile = "EyeRSmile"
ParamBrowLY = "BrowLY"
ParamBrowRY = "BrowRY"
ParamMouthOpenY = "MouthOpenY"
ParamMouthForm = "MouthForm"
ParamCheek = "Cheek"
```

The `param_map` maps **rig param ID → template param name**. `remap()` inverts this: given template output `{"AngleX": -5.2}`, returns `{"ParamAngleX": -5.2}`.

Unmapped template params are dropped (the rig doesn't have them).
Unmapped rig params are unaffected (driven by other sources like BCI, physics).

### 5. `mlp/infer.py` update

Replace the current `Predictor` (hardcoded to hiyori_v2, 956-d input):

```python
class Predictor:
    def __init__(self, template: Template, device: str = "cpu"):
        # Load model from template.model_path
        # Use checkpoint's input_dim/n_params (not RigConfig)
        # Store template.curves for post-processing

    def predict(self, features: np.ndarray) -> dict[str, float]:
        # features: (58,) or (1014,) depending on checkpoint
        # Returns: {template_param_name: value}

    def predict_with_curves(self, features: np.ndarray) -> dict[str, float]:
        # predict() + apply response curves
```

Keep backward compat by keeping the old `load_predictor()` working (deprecated).

### 6. `manifests/hiyori.toml` (new file)

First manifest, mapping Hiyori's VTS param names to humanoid-anime template names. This validates the system works end-to-end.

## Data Flow

```
Webcam → MediaPipe → 58-d features
    → Template("humanoid-anime").model → 13 template params
    → Template.curves.apply() → curved params
    → Manifest("hiyori").remap() → rig param names (ParamAngleX, etc.)
    → VTS WebSocket (Muse bridge handles this part)
```

## What This Does NOT Include

- Template auto-discovery or registry (load by name, one template)
- CLI for listing/validating templates
- Additional templates beyond humanoid-anime
- Schema version migration tooling
- Runtime webcam → VTS wiring (Phase 6, belongs in Muse bridge)

## Files Changed/Created

| File | Action |
|---|---|
| `templates/humanoid-anime/curves.toml` | Create |
| `templates/humanoid-anime/model.pt` | Copy from checkpoints |
| `templates/loader.py` | Create |
| `templates/__init__.py` | Create (empty or re-export) |
| `rig/manifest.py` | Create |
| `manifests/hiyori.toml` | Create |
| `mlp/infer.py` | Update (template-aware predictor) |

## Success Criteria

1. `load_template("humanoid-anime")` returns a Template with schema, model, and curves
2. `load_manifest("manifests/hiyori.toml")` returns a Manifest
3. End-to-end: random 58-d input → Predictor → curves → manifest remap → dict with `ParamAngleX` etc. keys
4. Existing training pipeline unaffected (schema.toml, verbs.toml, generate_verb_samples still work)
