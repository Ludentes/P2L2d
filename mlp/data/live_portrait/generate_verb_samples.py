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

from .renderer import SourceState, VerbRenderer
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
feature_dim_FULL = N_LANDMARKS * 2 + N_BLENDSHAPES + POSE_DIM  # 1014
feature_dim_BS = N_BLENDSHAPES + POSE_DIM  # 58 (blendshapes + pose only)


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


def extract_features(
    landmarker, mp_module, img_rgb: np.ndarray, *, bs_only: bool = False,
) -> np.ndarray | None:
    """Return feature vector, or None if no face detected.

    If bs_only=True, returns 58-d (blendshapes + pose).
    Otherwise returns 1014-d (landmarks + blendshapes + pose).
    """
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks or not result.face_blendshapes:
        return None

    # 52 ARKit blendshapes
    bs = result.face_blendshapes[0]
    bs_arr = np.array([b.score for b in bs], dtype=np.float32)
    if bs_arr.size != N_BLENDSHAPES:
        out = np.zeros(N_BLENDSHAPES, dtype=np.float32)
        out[:min(len(bs_arr), N_BLENDSHAPES)] = bs_arr[:N_BLENDSHAPES]
        bs_arr = out

    # 4x4 facial transformation matrix → rotation (deg) + translation
    mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
    pose = pose_from_matrix(mat)  # (6,)

    if bs_only:
        return np.concatenate([bs_arr, pose]).astype(np.float32)

    # 478 landmarks × (x, y) normalized ∈ [0, 1]
    lm = result.face_landmarks[0]
    lm_xy = np.empty(N_LANDMARKS * 2, dtype=np.float32)
    for i, p in enumerate(lm):
        lm_xy[2 * i] = p.x
        lm_xy[2 * i + 1] = p.y

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

def measure_source_baselines(
    renderer: VerbRenderer,
    landmarker,
    mp_module,
    sources: list[SourceState],
) -> list[tuple[float, float, float]]:
    """Render each source with neutral sliders, extract its MediaPipe pose.
    Returns list of (pitch_deg, yaw_deg, roll_deg) baselines.

    Used to bias rotate_pitch/yaw/roll jitter so the MediaPipe-measured
    AngleX/Y/Z labels land near 0 on average (not on the reference's
    baseline pose).
    """
    from .verb_sliders import VerbSliders as _VS
    neutral = _VS()
    baselines: list[tuple[float, float, float]] = []
    for i, src in enumerate(sources):
        img = renderer.render(src, neutral)
        feat = extract_features(landmarker, mp_module, img)
        if feat is None:
            # Source failed neutral detection; use zero baseline as fallback
            print(f"  WARN: source {i} has no face at neutral — using zero baseline")
            baselines.append((0.0, 0.0, 0.0))
            continue
        # Pose is always the last 6 dims regardless of feature mode
        rx = float(feat[-6])
        ry = float(feat[-5])
        rz = float(feat[-4])
        baselines.append((rx, ry, rz))
    return baselines


def generate(
    renderer: VerbRenderer,
    landmarker,
    mp_module,
    sources: list[SourceState],
    verbs: list[VerbEntry],
    schema: TemplateSchema,
    n_samples: int,
    seed: int = 0,
    source_baselines: list[tuple[float, float, float]] | None = None,
    bs_only: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    default_label = schema.default_label()
    idx_angle_x = schema.index_of("AngleX")
    idx_angle_y = schema.index_of("AngleY")
    idx_angle_z = schema.index_of("AngleZ")

    fdim = feature_dim_BS if bs_only else feature_dim_FULL
    features = np.empty((n_samples, fdim), dtype=np.float32)
    labels = np.empty((n_samples, schema.dim), dtype=np.float32)
    verb_names: list[str] = []
    source_ids = np.empty(n_samples, dtype=np.int32)
    rejected = 0
    t0 = time.time()

    i = 0
    while i < n_samples:
        src_id = int(rng.integers(0, len(sources)))
        source = sources[src_id]
        verb = verbs[int(rng.integers(0, len(verbs)))]
        sliders = jitter_sliders(verb.sliders, rng)
        # Bias rotation jitter by the negative of this source's baseline pose,
        # so the MediaPipe-measured labels center near 0° on average.
        if source_baselines is not None:
            b_rx, b_ry, b_rz = source_baselines[src_id]
            sliders.rotate_pitch -= b_rx  # rx/pitch
            sliders.rotate_yaw   -= b_ry  # ry/yaw
            sliders.rotate_roll  -= b_rz  # rz/roll
        img = renderer.render(source, sliders)
        feat = extract_features(landmarker, mp_module, img, bs_only=bs_only)
        if feat is None:
            rejected += 1
            continue

        # Build label: defaults + verb overrides + pose from MediaPipe
        label = schema.apply_verb_params(default_label, verb.params)
        # Pose lives at the end of the feature vector (last 6 dims)
        pose_offset = N_BLENDSHAPES if bs_only else (N_LANDMARKS * 2 + N_BLENDSHAPES)
        label[idx_angle_x] = feat[pose_offset + 1]  # ry (yaw) → horizontal head rotation
        label[idx_angle_y] = feat[pose_offset + 0]  # rx (pitch) → vertical head rotation
        label[idx_angle_z] = feat[pose_offset + 2]  # rz (roll) → head tilt

        features[i] = feat
        labels[i] = label
        verb_names.append(verb.name)
        source_ids[i] = src_id
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
        "source_ids": source_ids,
        "param_names": np.array(schema.names),
        "rejected": rejected,
        "seconds": dt,
    }


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _expand_references(path: Path) -> list[Path]:
    """Return all image paths. Accepts a file or a directory."""
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if path.is_file():
        return [path]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbs", type=Path, default=_DEFAULT_VERBS)
    ap.add_argument("--schema", type=Path, default=_DEFAULT_SCHEMA)
    ap.add_argument("--reference", type=Path, default=_DEFAULT_REF)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs-only", action="store_true",
                    help="Use blendshapes+pose (58-d) instead of full landmarks (1014-d)")
    args = ap.parse_args()

    fdim = feature_dim_BS if args.bs_only else feature_dim_FULL
    print(f"Schema:    {args.schema}")
    print(f"Verbs:     {args.verbs}")
    print(f"Reference: {args.reference}")
    print(f"Out:       {args.out}")
    print(f"N samples: {args.n}")
    print(f"Features:  {'blendshapes+pose (58-d)' if args.bs_only else 'full (1014-d)'}")

    schema = load_schema(args.schema)
    verbs = load_verbs(args.verbs)
    print(f"  {len(verbs)} verbs, {schema.dim} template params, feature_dim={fdim}")

    # Reference can be a single file or a directory of images
    ref_paths = _expand_references(args.reference)
    if not ref_paths:
        print(f"ERROR: no reference images at {args.reference}", file=sys.stderr)
        return 2
    print(f"References: {len(ref_paths)} image(s)")

    print("Loading LivePortrait...")
    renderer = VerbRenderer.from_default_checkpoints()

    sources: list = []
    source_paths: list[Path] = []
    for p in ref_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"  WARN: skipping unreadable {p}", file=sys.stderr)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            sources.append(renderer.precompute_source(img_rgb))
            source_paths.append(p)
        except RuntimeError as e:
            print(f"  WARN: skipping {p.name} ({e})", file=sys.stderr)
    if not sources:
        print("ERROR: no usable reference images", file=sys.stderr)
        return 2
    print(f"  precomputed {len(sources)} source(s)")

    print("Loading MediaPipe FaceLandmarker...")
    import mediapipe as mp_module
    landmarker = build_landmarker()

    print("Measuring per-source baseline poses...")
    baselines = measure_source_baselines(renderer, landmarker, mp_module, sources)
    # Drop sources whose neutral render did not produce a face (baseline==(0,0,0) fallback)
    kept_sources: list = []
    kept_paths: list[Path] = []
    kept_baselines: list[tuple[float, float, float]] = []
    for src, path, b in zip(sources, source_paths, baselines):
        if b == (0.0, 0.0, 0.0):
            print(f"  DROP {path.name}: neutral face detection failed")
            continue
        kept_sources.append(src)
        kept_paths.append(path)
        kept_baselines.append(b)
        print(f"  KEEP {path.name:12s} baseline pitch={b[0]:+6.2f}° yaw={b[1]:+6.2f}° roll={b[2]:+6.2f}°")
    if not kept_sources:
        print("ERROR: no usable sources after baseline check", file=sys.stderr)
        return 2
    sources = kept_sources
    source_paths = kept_paths
    baselines = kept_baselines

    print("\n--- Generating ---")
    out = generate(
        renderer, landmarker, mp_module, sources, verbs, schema, args.n,
        seed=args.seed, source_baselines=baselines, bs_only=args.bs_only,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=out["features"],
        labels=out["labels"],
        verb_names=out["verb_names"],
        source_ids=out["source_ids"],
        param_names=out["param_names"],
        source_paths=np.array([str(p) for p in source_paths]),
    )
    print(f"\nWrote {args.out}")
    print(f"  features={out['features'].shape}, labels={out['labels'].shape}")
    print(f"  rejected={out['rejected']}, {out['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
