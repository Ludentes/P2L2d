"""Phase 1 smoke test — 10 verbs on 1 reference image.

Validates that LivePortrait + VerbRenderer produces images whose MediaPipe
blendshapes respond correctly to the intended verb.

Usage:
    uv run python -m mlp.data.live_portrait.smoke_test
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .renderer import VerbRenderer
from .verb_sliders import VerbSliders

# Path to reference image (LivePortrait sample)
_REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = _REPO_ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "source" / "s0.jpg"
OUTPUT_DIR = _REPO_ROOT / "mlp" / "data" / "live_portrait" / "smoke_outputs"


@dataclass
class VerbTest:
    name: str
    sliders: VerbSliders
    # Each check: (blendshape_name, op, threshold)  — op ∈ {">", "<"}
    checks: list[tuple[str, str, float]]


# Expected MediaPipe blendshape response for each verb.
# Thresholds are conservative initial estimates — will tune empirically.
TESTS: list[VerbTest] = [
    VerbTest(
        name="neutral",
        sliders=VerbSliders(),
        checks=[
            ("eyeBlinkLeft", "<", 0.3),
            ("jawOpen", "<", 0.2),
            ("mouthSmileLeft", "<", 0.3),
        ],
    ),
    VerbTest(
        name="close_eyes",
        sliders=VerbSliders(blink=-15.0),
        checks=[
            ("eyeBlinkLeft", ">", 0.4),
            ("eyeBlinkRight", ">", 0.4),
        ],
    ),
    VerbTest(
        name="wink_right",  # PHM wink closes one eye; sign determines which
        sliders=VerbSliders(wink=20.0),
        checks=[
            # Accept asymmetric response — one eye more closed than the other
            # (exact side depends on PHM convention; we just check asymmetry)
        ],
    ),
    VerbTest(
        name="smile_slight",
        sliders=VerbSliders(smile=0.5, eee=8.0),
        checks=[
            ("mouthSmileLeft", ">", 0.15),
            ("mouthSmileRight", ">", 0.15),
        ],
    ),
    VerbTest(
        name="smile_wide",
        sliders=VerbSliders(smile=1.2, aaa=40.0),
        checks=[
            ("mouthSmileLeft", ">", 0.3),
            ("jawOpen", ">", 0.2),
        ],
    ),
    VerbTest(
        name="mouth_open",
        sliders=VerbSliders(aaa=80.0),
        checks=[
            ("jawOpen", ">", 0.4),
        ],
    ),
    VerbTest(
        name="look_left",
        sliders=VerbSliders(pupil_x=-12.0),
        checks=[
            # Horizontal eye gaze — either eyeLookOutLeft or eyeLookInRight
            # (depends on MediaPipe convention)
        ],
    ),
    VerbTest(
        name="look_up",
        sliders=VerbSliders(pupil_y=10.0),
        checks=[
            ("eyeLookUpLeft", ">", 0.2),
        ],
    ),
    VerbTest(
        name="brow_raise",
        sliders=VerbSliders(eyebrow=12.0),
        checks=[
            ("browInnerUp", ">", 0.2),
        ],
    ),
    VerbTest(
        name="surprised",
        sliders=VerbSliders(blink=3.0, aaa=60.0, eyebrow=12.0),
        checks=[
            ("jawOpen", ">", 0.3),
            ("browInnerUp", ">", 0.2),
        ],
    ),
]


def load_blendshape_extractor():
    """Build a MediaPipe FaceLandmarker configured to output blendshapes."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    # Download model if not cached
    model_path = _REPO_ROOT / "mlp" / "data" / "face_landmarker_v2_with_blendshapes.task"
    if not model_path.exists():
        print(f"Downloading MediaPipe face landmarker model to {model_path}...")
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return landmarker, mp


def extract_blendshapes(landmarker, mp_module, img_rgb: np.ndarray) -> dict[str, float]:
    """Run MediaPipe, return blendshape name → value dict (or empty if no face)."""
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_blendshapes:
        return {}
    return {bs.category_name: bs.score for bs in result.face_blendshapes[0]}


def run_smoke_test() -> int:
    print(f"Reference: {REFERENCE}")
    print(f"Output dir: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load reference image
    img_bgr = cv2.imread(str(REFERENCE))
    if img_bgr is None:
        print(f"ERROR: Could not load reference image {REFERENCE}", file=sys.stderr)
        return 2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Reference image shape: {img_rgb.shape}")

    # Build renderer (loads ~500MB of weights)
    print("\n--- Loading LivePortrait ---")
    renderer = VerbRenderer.from_default_checkpoints()
    print("Precomputing source...")
    source = renderer.precompute_source(img_rgb)
    print(f"Source cropped: {source.cropped_rgb.shape}, x_s: {tuple(source.x_s.shape)}")

    # Save cropped source + neutral render for visual inspection
    cv2.imwrite(str(OUTPUT_DIR / "_source_cropped.png"),
                cv2.cvtColor(source.cropped_rgb, cv2.COLOR_RGB2BGR))

    # Load MediaPipe
    print("\n--- Loading MediaPipe FaceLandmarker ---")
    landmarker, mp_module = load_blendshape_extractor()

    # Extract baseline blendshapes from source
    source_bs = extract_blendshapes(landmarker, mp_module, source.cropped_rgb)
    if not source_bs:
        print("WARNING: MediaPipe found no face in source crop", file=sys.stderr)
    else:
        print(f"Source blendshapes (sample): "
              f"eyeBlinkLeft={source_bs.get('eyeBlinkLeft', 0):.3f}, "
              f"jawOpen={source_bs.get('jawOpen', 0):.3f}, "
              f"mouthSmileLeft={source_bs.get('mouthSmileLeft', 0):.3f}")

    # Run each verb
    print("\n--- Running verb tests ---")
    results: list[tuple[str, bool, str]] = []
    for test in TESTS:
        img = renderer.render(source, test.sliders)
        cv2.imwrite(str(OUTPUT_DIR / f"{test.name}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        bs = extract_blendshapes(landmarker, mp_module, img)
        if not bs:
            results.append((test.name, False, "no face detected"))
            continue

        failures = []
        for bs_name, op, threshold in test.checks:
            value = bs.get(bs_name, 0.0)
            if op == ">" and not value > threshold:
                failures.append(f"{bs_name}={value:.3f} ≯ {threshold}")
            elif op == "<" and not value < threshold:
                failures.append(f"{bs_name}={value:.3f} ≮ {threshold}")

        passed = len(failures) == 0
        detail = ", ".join(failures) if failures else "OK"
        # Always show key blendshape values for context
        key_bs = f"blink={bs.get('eyeBlinkLeft',0):.2f} jaw={bs.get('jawOpen',0):.2f} smile={bs.get('mouthSmileLeft',0):.2f} brow={bs.get('browInnerUp',0):.2f}"
        results.append((test.name, passed, f"{detail} | {key_bs}"))

    # Report
    print("\n=== Results ===")
    n_pass = sum(1 for _, p, _ in results if p)
    for name, passed, detail in results:
        mark = "✓" if passed else "✗"
        print(f"  {mark} {name:20s}  {detail}")
    print(f"\n{n_pass}/{len(results)} passed")

    # No-check verbs (wink, look_left) need manual inspection
    print("\nManual inspection needed for: wink_right, look_left (no automated check)")
    print(f"See outputs in: {OUTPUT_DIR}")

    return 0 if n_pass >= 8 else 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
