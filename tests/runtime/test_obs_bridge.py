"""Unit tests for OBS bridge — no OBS or EEG connection required."""
import time

import pytest

from runtime.obs_bridge import ObsBridge


def _bridge() -> ObsBridge:
    return ObsBridge()


def test_initial_settings_are_neutral():
    b = _bridge()
    s = b._compute_settings()
    assert s["brightness"] == 0.0
    assert s["hue_shift"] == 0.0
    # saturation at 0 concentration/relaxation → 1.5 - 0 = 1.5
    assert s["saturation"] == 1.5


def test_relaxation_reduces_saturation():
    b = _bridge()
    b._relaxation = 1.0
    s = b._compute_settings()
    assert s["saturation"] == 0.0


def test_concentration_increases_hue_shift():
    b = _bridge()
    b._concentration = 1.0
    s = b._compute_settings()
    assert s["hue_shift"] == 25.0


def test_clench_pulse_sets_brightness():
    b = _bridge()
    b._clench_until = time.time() + 10.0  # active pulse
    s = b._compute_settings()
    assert s["brightness"] == 0.3


def test_expired_clench_brightness_zero():
    b = _bridge()
    b._clench_until = time.time() - 1.0  # expired
    s = b._compute_settings()
    assert s["brightness"] == 0.0


def test_ingest_brain_metrics():
    b = _bridge()
    b._ingest({
        "type": "metrics",
        "brain": {"concentration": 0.7, "relaxation": 0.4},
        "imu": {},
    })
    assert b._concentration == pytest.approx(0.7)
    assert b._relaxation == pytest.approx(0.4)


def test_ingest_jaw_clench_sets_pulse():
    b = _bridge()
    before = time.time()
    b._ingest({
        "type": "metrics",
        "imu": {"jaw_clench": True},
    })
    assert b._clench_until > before


def test_ingest_ignores_non_metrics():
    b = _bridge()
    b._ingest({"type": "eeg_raw", "data": []})
    assert b._concentration == 0.0
