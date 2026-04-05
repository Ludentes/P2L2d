"""Generate verb-labeled training samples for the MLP.

Pipeline per sample:
  1. Pick a random verb from verbs.toml
  2. Jitter its sliders slightly (expression intensity variation)
  3. Add random head pose rotation (rotate_pitch/yaw/roll)
  4. Render via LivePortrait
  5. Extract MediaPipe features: 478×2 landmarks + 52 blendshapes + 6 pose = 1014-d
  6. Build label: template defaults + verb.params override + MediaPipe pose → AngleX/Y/Z

Output: NPZ with
  features       (N, 1014) float32
  labels         (N, P)    float32  — P = schema.dim
  verb_names     (N,)      str
  landmark_ok    (N,)      bool     — whether MediaPipe found a face
  param_names    (P,)      str      — for reference

Usage:
    uv run python -m mlp.data.live_portrait.generate_verb_samples \\
        --n 200 --out mlp/data/live_portrait/datasets/dev_200.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .renderer import VerbRenderer
from .template_schema import TemplateSchema, load_schema
from .verb_library import VerbEntry, load_verbs
from .verb_sliders import VerbSliders

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_VERBS = _REPO_ROOT / "templates" / "humanoid-anime" / "verbs.toml"
_DEFAULT_SCHEMA = _REPO_ROOT / "templates" / "humanoid-anime" / "schema.toml"
_DEFAULT_REF = _REPO_ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "source" / "s0.jpg"
_DEFAULT_OUT = _REPO_ROOT / "mlp" / "data" / "live_portrait" / "datasets" / "dev_200.npz"

N_LANDMARKS = 478
N_BLENDSHAPES = 52
POSE_DIM = 6  # rx, ry, rz (deg), tx, ty, tz
FEATURE_DIM = N_LANDMARKS * 2 + N_BLENDSHAPES + POSE_DIM  # 1014


# -------------------- MediaPipe --------------------

def build_landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = _REPO_ROOT / "mlp" / "data" / "face_landmarker_v2_with_blendshapes.task"
    if not model_path.exists():
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
    return vision.FaceLandmarker.create_from_options(options)


def extract_features(landmarker, mp_module, img_rgb: np.ndarray) -> np.ndarray | None:
    """Return 1014-d feature vector, or None if no face detected."""
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks or not result.face_blendshapes:
        return None

    # 478 landmarks × (x, y) normalized ∈ [0, 1]
    lm = result.face_landmarks[0]
    lm_xy = np.empty(N_LANDMARKS * 2, dtype=np.float32)
    for i, p in enumerate(lm):
        lm_xy[2 * i] = p.x
        lm_xy[2 * i + 1] = p.y

    # 52 ARKit blendshapes
    bs = result.face_blendshapes[0]
    bs_arr = np.array([b.score for b in bs], dtype=np.float32)
    if bs_arr.size != N_BLENDSHAPES:
        # In rare edge cases MP may return a different count — pad/truncate
        out = np.zeros(N_BLENDSHAPES, dtype=np.float32)
        out[:min(len(bs_arr), N_BLENDSHAPES)] = bs_arr[:N_BLENDSHAPES]
        bs_arr = out

    # 4x4 facial transformation matrix → rotation (deg) + translation
    mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
    pose = pose_from_matrix(mat)  # (6,)

    return np.concatenate([lm_xy, bs_arr, pose]).astype(np.float32)


def pose_from_matrix(mat: np.ndarray) -> np.ndarray:
    """Extract (rx_deg, ry_deg, rz_deg, tx, ty, tz) from a 4x4 transform matrix."""
    R = mat[:3, :3]
    t = mat[:3, 3]
    # Standard XYZ-euler extraction (not pinned to any rig convention — MLP learns mapping)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array(
        [np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz), t[0], t[1], t[2]],
        dtype=np.float32,
    )


# -------------------- Jitter --------------------

_SLIDER_JITTER_FRAC = 0.15  # ±15% of the slider value
_POSE_JITTER = {
    "rotate_pitch": 12.0,
    "rotate_yaw": 15.0,
    "rotate_roll": 8.0,
}


def jitter_sliders(base: VerbSliders, rng: np.random.Generator) -> VerbSliders:
    """Slightly perturb slider values around the verb's authored intensities."""
    kwargs = {}
    for field, val in base.as_dict().items():
        if field.startswith("rotate_"):
            # Pose sliders are handled separately
            kwargs[field] = val
            continue
        if val == 0.0:
            kwargs[field] = 0.0
        else:
            kwargs[field] = float(val * (1.0 + rng.uniform(-_SLIDER_JITTER_FRAC, _SLIDER_JITTER_FRAC)))
    jittered = replace(base, **kwargs)
    # Apply pose jitter independently
    jittered.rotate_pitch += float(rng.uniform(-_POSE_JITTER["rotate_pitch"], _POSE_JITTER["rotate_pitch"]))
    jittered.rotate_yaw   += float(rng.uniform(-_POSE_JITTER["rotate_yaw"],   _POSE_JITTER["rotate_yaw"]))
    jittered.rotate_roll  += float(rng.uniform(-_POSE_JITTER["rotate_roll"],  _POSE_JITTER["rotate_roll"]))
    return jittered


# -------------------- Generation loop --------------------

def generate(
    renderer: VerbRenderer,
    landmarker,
    mp_module,
    source,
    verbs: list[VerbEntry],
    schema: TemplateSchema,
    n_samples: int,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    default_label = schema.default_label()
    idx_angle_x = schema.index_of("AngleX")
    idx_angle_y = schema.index_of("AngleY")
    idx_angle_z = schema.index_of("AngleZ")

    features = np.empty((n_samples, FEATURE_DIM), dtype=np.float32)
    labels = np.empty((n_samples, schema.dim), dtype=np.float32)
    verb_names: list[str] = []
    rejected = 0
    t0 = time.time()

    i = 0
    while i < n_samples:
        verb = verbs[int(rng.integers(0, len(verbs)))]
        sliders = jitter_sliders(verb.sliders, rng)
        img = renderer.render(source, sliders)
        feat = extract_features(landmarker, mp_module, img)
        if feat is None:
            rejected += 1
            continue

        # Build label: defaults + verb overrides + pose from MediaPipe
        label = schema.apply_verb_params(default_label, verb.params)
        # feat's pose is at indices [N_LANDMARKS*2 + N_BLENDSHAPES : ...]
        pose_offset = N_LANDMARKS * 2 + N_BLENDSHAPES
        label[idx_angle_x] = feat[pose_offset + 1]  # ry (yaw) → horizontal head rotation
        label[idx_angle_y] = feat[pose_offset + 0]  # rx (pitch) → vertical head rotation
        label[idx_angle_z] = feat[pose_offset + 2]  # rz (roll) → head tilt

        features[i] = feat
        labels[i] = label
        verb_names.append(verb.name)
        i += 1

        if (i % 50) == 0:
            dt = time.time() - t0
            rate = i / dt
            print(f"  {i}/{n_samples}  ({rate:.1f}/s, rejected={rejected})")

    dt = time.time() - t0
    print(f"Done: {n_samples} samples in {dt:.1f}s  ({n_samples/dt:.1f}/s, rejected={rejected})")

    return {
        "features": features,
        "labels": labels,
        "verb_names": np.array(verb_names),
        "param_names": np.array(schema.names),
        "rejected": rejected,
        "seconds": dt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbs", type=Path, default=_DEFAULT_VERBS)
    ap.add_argument("--schema", type=Path, default=_DEFAULT_SCHEMA)
    ap.add_argument("--reference", type=Path, default=_DEFAULT_REF)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Schema:    {args.schema}")
    print(f"Verbs:     {args.verbs}")
    print(f"Reference: {args.reference}")
    print(f"Out:       {args.out}")
    print(f"N samples: {args.n}")

    schema = load_schema(args.schema)
    verbs = load_verbs(args.verbs)
    print(f"  {len(verbs)} verbs, {schema.dim} template params, feature_dim={FEATURE_DIM}")

    img_bgr = cv2.imread(str(args.reference))
    if img_bgr is None:
        print(f"ERROR: cannot read {args.reference}", file=sys.stderr)
        return 2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print("Loading LivePortrait...")
    renderer = VerbRenderer.from_default_checkpoints()
    source = renderer.precompute_source(img_rgb)

    print("Loading MediaPipe FaceLandmarker...")
    import mediapipe as mp_module
    landmarker = build_landmarker()

    print("\n--- Generating ---")
    out = generate(renderer, landmarker, mp_module, source, verbs, schema, args.n, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=out["features"],
        labels=out["labels"],
        verb_names=out["verb_names"],
        param_names=out["param_names"],
    )
    print(f"\nWrote {args.out}")
    print(f"  features={out['features'].shape}, labels={out['labels'].shape}")
    print(f"  rejected={out['rejected']}, {out['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
