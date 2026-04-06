# Template + Manifest System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle the trained MLP, schema, and response curves into a self-contained template with a manifest system for per-rig param remapping.

**Architecture:** Template directory (`templates/humanoid-anime/`) packages schema + model + curves. A manifest TOML maps any rig's custom param names to template canonical names. Updated `Predictor` loads from templates and returns template param names; manifest remaps to rig names at runtime.

**Tech Stack:** Python 3.12, TOML (tomllib), PyTorch (checkpoint loading), numpy

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `templates/loader.py` | Create | `Template` dataclass + `load_template()` — loads schema, model path, curves from template dir |
| `templates/__init__.py` | Create | Re-export `load_template`, `Template` |
| `templates/humanoid-anime/curves.toml` | Create | Per-param response curves (materialized from `HIYORI_DEFAULTS`) |
| `templates/humanoid-anime/model.pt` | Copy | Trained checkpoint (from `mlp/checkpoints/humanoid-anime-bs58/`) |
| `rig/manifest.py` | Create | `Manifest` dataclass + `load_manifest()` — loads TOML, remaps params |
| `manifests/hiyori.toml` | Create | First manifest — maps Hiyori VTS param names to template names |
| `mlp/infer.py` | Modify | Add `TemplatePredictor` class that loads from `Template`, supports 58-d input |
| `tests/templates/__init__.py` | Create | Test package |
| `tests/templates/test_loader.py` | Create | Tests for template loading |
| `tests/rig/test_manifest.py` | Create | Tests for manifest loading + remapping |
| `tests/mlp/test_infer.py` | Modify | Add tests for `TemplatePredictor` |

---

### Task 1: Template curves.toml

**Files:**
- Create: `templates/humanoid-anime/curves.toml`

- [ ] **Step 1: Create curves.toml from existing HIYORI_DEFAULTS**

The `ResponseCurveSet.from_toml()` expects format `[params.<name>]`. Use template param names (no `Param` prefix) since these are template-space curves.

```toml
# templates/humanoid-anime/curves.toml
# Post-MLP response curves for humanoid-anime template.
# Applied at inference time, not baked into the model.
# Param names are template-canonical (matching schema.toml).

[params.EyeLOpen]
type = "gamma"
gamma = 0.7
lo = 0.0
hi = 1.0

[params.EyeROpen]
type = "gamma"
gamma = 0.7
lo = 0.0
hi = 1.0

[params.EyeLSmile]
type = "gamma"
gamma = 0.8
lo = 0.0
hi = 1.0

[params.EyeRSmile]
type = "gamma"
gamma = 0.8
lo = 0.0
hi = 1.0

[params.MouthOpenY]
type = "gamma"
gamma = 0.6
lo = 0.0
hi = 1.0

[params.MouthForm]
type = "linear_scale"
scale = 1.3
centre = 0.0
lo = -1.0
hi = 1.0

[params.BrowLY]
type = "linear_scale"
scale = 1.2
centre = 0.0
lo = -1.0
hi = 1.0

[params.BrowRY]
type = "linear_scale"
scale = 1.2
centre = 0.0
lo = -1.0
hi = 1.0
```

Note: `MouthForm` range is `[-1.0, 1.0]` here (matching `schema.toml`), not `[-2.0, 1.0]` as in `HIYORI_DEFAULTS` which was Hiyori-specific. The template curves use the template's canonical range.

- [ ] **Step 2: Commit**

```bash
git add templates/humanoid-anime/curves.toml
git commit -m "feat(template): add curves.toml for humanoid-anime"
```

---

### Task 2: Copy model.pt to template directory

**Files:**
- Create: `templates/humanoid-anime/model.pt` (copy from checkpoints)

- [ ] **Step 1: Copy the trained checkpoint**

```bash
cp mlp/checkpoints/humanoid-anime-bs58/model.pt templates/humanoid-anime/model.pt
```

- [ ] **Step 2: Verify the checkpoint loads**

```bash
uv run python -c "
import torch
ckpt = torch.load('templates/humanoid-anime/model.pt', weights_only=False, map_location='cpu')
print(f'input_dim={ckpt[\"input_dim\"]}, n_params={ckpt[\"n_params\"]}')
print(f'params: {ckpt[\"param_names\"]}')
"
```

Expected: `input_dim=58, n_params=13` and the 13 param names.

- [ ] **Step 3: Add model.pt to .gitignore (it's a binary)**

Append to `.gitignore`:
```
templates/*/model.pt
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore template model.pt binaries"
```

---

### Task 3: Template loader

**Files:**
- Create: `templates/__init__.py`
- Create: `templates/loader.py`
- Create: `tests/templates/__init__.py`
- Create: `tests/templates/test_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/templates/__init__.py
# (empty)
```

```python
# tests/templates/test_loader.py
"""Tests for templates.loader — template discovery and loading."""
from pathlib import Path

import pytest


def test_load_template_returns_template():
    from templates.loader import load_template, Template

    t = load_template("humanoid-anime")
    assert isinstance(t, Template)
    assert t.name == "humanoid-anime"


def test_template_has_schema():
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    assert t.schema.dim == 13
    assert t.schema.names[0] == "AngleX"


def test_template_has_model_path():
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    # model_path may or may not exist (gitignored), but path should be set
    assert t.model_path.name == "model.pt"
    assert "humanoid-anime" in str(t.model_path)


def test_template_has_curves():
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    assert "EyeLOpen" in t.curves.curves


def test_load_nonexistent_template_raises():
    from templates.loader import load_template

    with pytest.raises(FileNotFoundError):
        load_template("nonexistent-template")


def test_template_without_curves_gets_empty_set():
    """If curves.toml is missing, template should still load with identity curves."""
    import tempfile
    import shutil
    from templates.loader import load_template, _TEMPLATES_DIR

    # Create a minimal temp template
    tmp = _TEMPLATES_DIR / "_test_minimal"
    tmp.mkdir(exist_ok=True)
    try:
        # Copy just schema.toml
        shutil.copy(_TEMPLATES_DIR / "humanoid-anime" / "schema.toml", tmp / "schema.toml")
        t = load_template("_test_minimal")
        assert len(t.curves.curves) == 0  # no curves = identity
    finally:
        shutil.rmtree(tmp)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/templates/test_loader.py -v
```

Expected: FAIL — `templates.loader` module doesn't exist.

- [ ] **Step 3: Implement templates/loader.py**

```python
# templates/loader.py
"""Template loader — loads schema, model path, and response curves from a template directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlp.curves import ResponseCurveSet
from mlp.data.live_portrait.template_schema import TemplateSchema, load_schema

_TEMPLATES_DIR = Path(__file__).parent


@dataclass
class Template:
    """A loaded template: schema + model path + response curves."""

    name: str
    schema: TemplateSchema
    model_path: Path
    curves: ResponseCurveSet


def load_template(name: str) -> Template:
    """Load a template by name from the templates directory.

    Args:
        name: Template directory name (e.g. "humanoid-anime").

    Returns:
        A Template with schema, model path, and curves loaded.

    Raises:
        FileNotFoundError: If the template directory or schema.toml is missing.
    """
    template_dir = _TEMPLATES_DIR / name
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    schema_path = template_dir / "schema.toml"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.toml not found in {template_dir}")

    schema = load_schema(schema_path)
    model_path = template_dir / "model.pt"

    curves_path = template_dir / "curves.toml"
    curves = ResponseCurveSet.from_toml(curves_path) if curves_path.exists() else ResponseCurveSet()

    return Template(name=name, schema=schema, model_path=model_path, curves=curves)
```

```python
# templates/__init__.py
"""Template system — pre-trained archetypes for Live2D parameter prediction."""
from .loader import Template, load_template

__all__ = ["Template", "load_template"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/templates/test_loader.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add templates/__init__.py templates/loader.py tests/templates/__init__.py tests/templates/test_loader.py
git commit -m "feat(template): template loader with schema + model + curves"
```

---

### Task 4: Manifest loader

**Files:**
- Create: `rig/manifest.py`
- Create: `manifests/hiyori.toml`
- Create: `tests/rig/test_manifest.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/rig/test_manifest.py
"""Tests for rig.manifest — per-rig param name mapping."""
from pathlib import Path

import pytest


def test_load_manifest():
    from rig.manifest import load_manifest

    m = load_manifest(Path("manifests/hiyori.toml"))
    assert m.template_name == "humanoid-anime"
    assert "ParamAngleX" in m.param_map


def test_manifest_remap_forward():
    """Template output names → rig param names."""
    from rig.manifest import load_manifest

    m = load_manifest(Path("manifests/hiyori.toml"))
    template_output = {"AngleX": -5.2, "MouthOpenY": 0.73}
    rig_output = m.remap(template_output)
    assert rig_output == {"ParamAngleX": -5.2, "ParamMouthOpenY": 0.73}


def test_manifest_remap_drops_unmapped():
    """Template params not in any rig mapping are dropped."""
    from rig.manifest import Manifest

    m = Manifest(template_name="t", param_map={"ParamX": "TemplateX"})
    result = m.remap({"TemplateX": 1.0, "TemplateY": 2.0})
    assert result == {"ParamX": 1.0}
    assert "TemplateY" not in result


def test_manifest_remap_empty_input():
    from rig.manifest import Manifest

    m = Manifest(template_name="t", param_map={"ParamX": "TemplateX"})
    assert m.remap({}) == {}


def test_manifest_all_13_params_mapped():
    """Hiyori manifest should map all 13 template params."""
    from rig.manifest import load_manifest

    m = load_manifest(Path("manifests/hiyori.toml"))
    expected_template_params = [
        "AngleX", "AngleY", "AngleZ", "EyeLOpen", "EyeROpen",
        "EyeBallX", "EyeLSmile", "EyeRSmile", "BrowLY", "BrowRY",
        "MouthOpenY", "MouthForm", "Cheek",
    ]
    mapped_template_params = set(m.param_map.values())
    for p in expected_template_params:
        assert p in mapped_template_params, f"Template param {p!r} not mapped in manifest"


def test_load_manifest_missing_file():
    from rig.manifest import load_manifest

    with pytest.raises(FileNotFoundError):
        load_manifest(Path("manifests/nonexistent.toml"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/rig/test_manifest.py -v
```

Expected: FAIL — `rig.manifest` module doesn't exist.

- [ ] **Step 3: Create the hiyori manifest**

```toml
# manifests/hiyori.toml
# Manifest for Hiyori VTS rig → humanoid-anime template.
#
# Maps Hiyori's Live2D param IDs (left side) to the template's
# canonical param names (right side). At runtime, MLP outputs
# template params which get remapped to these rig-specific names
# before sending to VTube Studio.

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

- [ ] **Step 4: Implement rig/manifest.py**

```python
# rig/manifest.py
"""Per-rig manifest — maps user's custom Live2D param names to template canonical names."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Manifest:
    """Maps a rig's param names to a template's canonical param names.

    Attributes:
        template_name: Which template this rig uses (e.g. "humanoid-anime").
        param_map: Mapping of rig_param_id → template_param_name.
                   e.g. {"ParamAngleX": "AngleX", "ParamMouthOpenY": "MouthOpenY"}
    """

    template_name: str
    param_map: dict[str, str]  # rig_param → template_param

    def remap(self, template_output: dict[str, float]) -> dict[str, float]:
        """Convert template param names to rig param names.

        Given MLP output like {"AngleX": -5.2}, returns {"ParamAngleX": -5.2}
        using the inverse of param_map. Template params not mapped to any rig
        param are dropped.
        """
        # Build reverse map: template_param → rig_param
        reverse = {tp: rp for rp, tp in self.param_map.items()}
        return {
            reverse[tp]: value
            for tp, value in template_output.items()
            if tp in reverse
        }


def load_manifest(path: Path) -> Manifest:
    """Load a manifest from a TOML file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Manifest(
        template_name=data["template"],
        param_map=dict(data.get("param_map", {})),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/rig/test_manifest.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rig/manifest.py manifests/hiyori.toml tests/rig/test_manifest.py
git commit -m "feat(rig): manifest loader with param remapping"
```

---

### Task 5: TemplatePredictor in mlp/infer.py

**Files:**
- Modify: `mlp/infer.py`
- Modify: `tests/mlp/test_infer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/mlp/test_infer.py`:

```python
def test_template_predictor_output_keys():
    """TemplatePredictor returns template param names, not rig param names."""
    pytest.importorskip("torch")
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    if not t.model_path.exists():
        pytest.skip("template model.pt not present")

    from mlp.infer import TemplatePredictor

    p = TemplatePredictor(t)
    features = np.zeros(58, dtype=np.float32)
    result = p.predict(features)
    assert set(result.keys()) == set(t.schema.names)
    assert "AngleX" in result


def test_template_predictor_with_curves():
    """predict_with_curves applies response curves."""
    pytest.importorskip("torch")
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    if not t.model_path.exists():
        pytest.skip("template model.pt not present")

    from mlp.infer import TemplatePredictor

    p = TemplatePredictor(t)
    features = np.zeros(58, dtype=np.float32)
    raw = p.predict(features)
    curved = p.predict_with_curves(features)
    # Curved and raw may differ for params with non-identity curves
    assert set(curved.keys()) == set(raw.keys())


def test_template_predictor_accepts_batch():
    """TemplatePredictor handles (B, 58) input."""
    pytest.importorskip("torch")
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    if not t.model_path.exists():
        pytest.skip("template model.pt not present")

    from mlp.infer import TemplatePredictor

    p = TemplatePredictor(t)
    features = np.zeros((4, 58), dtype=np.float32)
    result = p.predict_batch(features)
    assert result.shape == (4, 13)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/mlp/test_infer.py::test_template_predictor_output_keys -v
```

Expected: FAIL — `TemplatePredictor` doesn't exist.

- [ ] **Step 3: Add TemplatePredictor to mlp/infer.py**

Add the following class to `mlp/infer.py` (keep existing `Predictor` class intact for backward compat):

```python
from templates.loader import Template


class TemplatePredictor:
    """Template-aware inference: loads model from template, returns template param names."""

    def __init__(self, template: Template, device: str = "cpu") -> None:
        self._template = template
        self._device = device
        ckpt = torch.load(template.model_path, weights_only=False, map_location=device)
        self._model = CartoonAliveMLP(
            n_params=ckpt["n_params"],
            input_dim=ckpt["input_dim"],
        )
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._model.to(device)
        self._param_names: list[str] = ckpt["param_names"]

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """Map feature vector → dict of {template_param_name: value}.

        Args:
            features: float32 array of shape (input_dim,).
        """
        x = torch.from_numpy(features.reshape(1, -1).astype(np.float32)).to(self._device)
        with torch.no_grad():
            y = self._model(x)
        values = y.squeeze(0).cpu().numpy()
        return dict(zip(self._param_names, values.tolist()))

    def predict_with_curves(self, features: np.ndarray) -> dict[str, float]:
        """predict() + apply template response curves."""
        raw = self.predict(features)
        raw_arr = np.array([raw[n] for n in self._param_names], dtype=np.float32)
        curved_arr = self._template.curves.apply(raw_arr, self._param_names)
        return dict(zip(self._param_names, curved_arr.tolist()))

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch inference. Returns (B, n_params) float32 array."""
        x = torch.from_numpy(features.astype(np.float32)).to(self._device)
        with torch.no_grad():
            y = self._model(x)
        return y.cpu().numpy()
```

Also add the import at the top of `mlp/infer.py`:

```python
from templates.loader import Template
```

- [ ] **Step 4: Run all infer tests**

```bash
uv run pytest tests/mlp/test_infer.py -v
```

Expected: All tests PASS (old Predictor tests may skip if hiyori_v2 checkpoint missing; new TemplatePredictor tests should pass).

- [ ] **Step 5: Commit**

```bash
git add mlp/infer.py tests/mlp/test_infer.py
git commit -m "feat(mlp): TemplatePredictor — template-aware inference with curves"
```

---

### Task 6: End-to-end integration test

**Files:**
- Create: `tests/test_end_to_end.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_end_to_end.py
"""End-to-end: features → template predictor → curves → manifest remap → rig params."""
from pathlib import Path

import numpy as np
import pytest


def test_full_pipeline():
    """58-d features → template MLP → curves → manifest → rig param dict."""
    pytest.importorskip("torch")
    from templates.loader import load_template
    from rig.manifest import load_manifest

    template = load_template("humanoid-anime")
    if not template.model_path.exists():
        pytest.skip("template model.pt not present")

    manifest = load_manifest(Path("manifests/hiyori.toml"))
    assert manifest.template_name == template.name

    from mlp.infer import TemplatePredictor

    predictor = TemplatePredictor(template)

    # Simulate a neutral face (zeros)
    features = np.zeros(58, dtype=np.float32)
    template_params = predictor.predict_with_curves(features)

    # Remap to rig params
    rig_params = manifest.remap(template_params)

    # All 13 rig params should be present
    assert len(rig_params) == 13
    assert "ParamAngleX" in rig_params
    assert "ParamMouthOpenY" in rig_params
    assert "ParamCheek" in rig_params

    # Values should be finite floats
    for name, value in rig_params.items():
        assert np.isfinite(value), f"{name} is not finite: {value}"
```

- [ ] **Step 2: Run the integration test**

```bash
uv run pytest tests/test_end_to_end.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: end-to-end template → manifest pipeline"
```

---

### Task 7: Final commit — bundle all Phase 5 work

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All tests pass (some may skip due to missing checkpoints).

- [ ] **Step 2: Push**

```bash
git push
```
