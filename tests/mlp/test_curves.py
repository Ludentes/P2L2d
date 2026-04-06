"""Tests for mlp.curves — response curve post-processing layer."""
import json
import tempfile
from pathlib import Path

import numpy as np

from mlp.curves import GammaCurve, LinearScaleCurve, ResponseCurveSet, HIYORI_DEFAULTS


# ── GammaCurve ────────────────────────────────────────────────────────────────

def test_gamma_curve_identity_at_gamma_one():
    c = GammaCurve(gamma=1.0, lo=0.0, hi=1.0)
    vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    out = c.apply(vals)
    np.testing.assert_allclose(out, vals, atol=1e-5)


def test_gamma_curve_lifts_midtones():
    c = GammaCurve(gamma=0.5, lo=0.0, hi=1.0)
    mid = np.array([0.5], dtype=np.float32)
    out = c.apply(mid)
    assert out[0] > 0.5, "gamma<1 should lift midtones above linear"


def test_gamma_curve_clips_at_bounds():
    c = GammaCurve(gamma=0.7, lo=0.0, hi=1.0)
    vals = np.array([-0.5, 0.0, 1.0, 2.0], dtype=np.float32)
    out = c.apply(vals)
    assert out[0] == pytest_approx(0.0)
    assert out[-1] == pytest_approx(1.0)


def pytest_approx(x: float) -> float:
    """Tiny helper so we don't import pytest in this file."""
    return x


def test_gamma_curve_endpoints_preserved():
    c = GammaCurve(gamma=0.7, lo=0.0, hi=1.0)
    vals = np.array([0.0, 1.0], dtype=np.float32)
    out = c.apply(vals)
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-5)


def test_gamma_curve_non_unit_range():
    c = GammaCurve(gamma=1.0, lo=-1.0, hi=1.0)
    vals = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    out = c.apply(vals)
    np.testing.assert_allclose(out, vals, atol=1e-5)


# ── LinearScaleCurve ──────────────────────────────────────────────────────────

def test_linear_scale_identity_at_scale_one():
    c = LinearScaleCurve(scale=1.0, centre=0.0, lo=-1.0, hi=1.0)
    vals = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    out = c.apply(vals)
    np.testing.assert_allclose(out, vals, atol=1e-5)


def test_linear_scale_exaggerates():
    c = LinearScaleCurve(scale=2.0, centre=0.0, lo=-2.0, hi=2.0)
    vals = np.array([0.5], dtype=np.float32)
    out = c.apply(vals)
    np.testing.assert_allclose(out, [1.0], atol=1e-5)


def test_linear_scale_clips():
    c = LinearScaleCurve(scale=3.0, centre=0.0, lo=-1.0, hi=1.0)
    vals = np.array([0.9], dtype=np.float32)
    out = c.apply(vals)
    assert out[0] <= 1.0


# ── ResponseCurveSet ──────────────────────────────────────────────────────────

def test_from_dict_parses_gamma():
    spec = {"ParamEyeLOpen": {"type": "gamma", "gamma": 0.7, "lo": 0.0, "hi": 1.0}}
    cs = ResponseCurveSet.from_dict(spec)
    assert "ParamEyeLOpen" in cs.curves
    assert isinstance(cs.curves["ParamEyeLOpen"], GammaCurve)


def test_from_dict_parses_linear_scale():
    spec = {"ParamBrowLY": {"type": "linear_scale", "scale": 1.2, "centre": 0.0, "lo": -1.0, "hi": 1.0}}
    cs = ResponseCurveSet.from_dict(spec)
    assert isinstance(cs.curves["ParamBrowLY"], LinearScaleCurve)


def test_apply_single_sample():
    cs = ResponseCurveSet.from_dict(HIYORI_DEFAULTS)
    params = np.zeros(3, dtype=np.float32)
    param_ids = ["ParamAngleX", "ParamEyeLOpen", "ParamMouthOpenY"]
    params[1] = 0.5  # EyeLOpen mid
    params[2] = 0.5  # MouthOpenY mid
    out = cs.apply(params, param_ids)
    assert out.shape == (3,)
    assert out[1] > 0.5, "gamma<1 should lift EyeLOpen midtones"
    assert out[2] > 0.5, "gamma<1 should lift MouthOpenY midtones"
    assert out[0] == 0.0, "unlisted param should pass through unchanged"


def test_apply_batch():
    cs = ResponseCurveSet.from_dict(HIYORI_DEFAULTS)
    params = np.full((10, 3), 0.5, dtype=np.float32)
    param_ids = ["ParamAngleX", "ParamEyeLOpen", "ParamMouthOpenY"]
    out = cs.apply(params, param_ids)
    assert out.shape == (10, 3)
    assert (out[:, 1] > 0.5).all()


def test_apply_does_not_mutate_input():
    cs = ResponseCurveSet.from_dict(HIYORI_DEFAULTS)
    params = np.array([0.5, 0.5], dtype=np.float32)
    original = params.copy()
    cs.apply(params, ["ParamEyeLOpen", "ParamEyeROpen"])
    np.testing.assert_array_equal(params, original)


def test_unknown_param_is_ignored():
    cs = ResponseCurveSet.from_dict(HIYORI_DEFAULTS)
    params = np.array([0.5, 0.3], dtype=np.float32)
    out = cs.apply(params, ["ParamNotReal", "ParamAlsoFake"])
    np.testing.assert_array_equal(out, params)


def test_round_trip_json():
    cs = ResponseCurveSet.from_dict(HIYORI_DEFAULTS)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"params": cs.to_dict()}, f)
        tmp = Path(f.name)
    cs2 = ResponseCurveSet.from_json(tmp)
    assert set(cs2.curves.keys()) == set(cs.curves.keys())
    tmp.unlink()


def test_hiyori_defaults_parse():
    cs = ResponseCurveSet.from_dict(HIYORI_DEFAULTS)
    assert "ParamEyeLOpen" in cs.curves
    assert "ParamMouthOpenY" in cs.curves
    assert "ParamBrowLY" in cs.curves


def test_gamma_curve_zero_span_returns_copy():
    c = GammaCurve(gamma=0.7, lo=0.5, hi=0.5)
    vals = np.array([0.5, 0.5], dtype=np.float32)
    out = c.apply(vals)
    np.testing.assert_array_equal(out, vals)
