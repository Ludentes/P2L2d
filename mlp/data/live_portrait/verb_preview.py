"""Render every verb in a verbs.toml as a labeled grid — for visual review.

Usage:
    uv run python -m mlp.data.live_portrait.verb_preview \\
        --verbs templates/humanoid-anime/verbs.toml \\
        --reference third_party/LivePortrait/assets/examples/source/s0.jpg \\
        --out mlp/data/live_portrait/smoke_outputs/verb_grid.png
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .renderer import VerbRenderer
from .verb_library import VerbEntry, load_verbs

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_VERBS = _REPO_ROOT / "templates" / "humanoid-anime" / "verbs.toml"
_DEFAULT_REF = _REPO_ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "source" / "s0.jpg"
_DEFAULT_OUT = _REPO_ROOT / "mlp" / "data" / "live_portrait" / "smoke_outputs" / "verb_grid.png"


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw text label on top of a tile (RGB uint8, in-place safe)."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (_, th), _ = cv2.getTextSize(text, font, scale, thickness)
    # Black bar at top
    cv2.rectangle(out, (0, 0), (out.shape[1], th + 8), (0, 0, 0), -1)
    cv2.putText(out, text, (4, th + 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def render_grid(
    renderer: VerbRenderer,
    verbs: list[VerbEntry],
    reference_rgb: np.ndarray,
    cols: int = 6,
) -> np.ndarray:
    """Render every verb on the reference image and assemble into a grid."""
    source = renderer.precompute_source(reference_rgb)

    tiles: list[np.ndarray] = []
    for verb in verbs:
        img = renderer.render(source, verb.sliders)
        tiles.append(_label(img, verb.name))

    rows = math.ceil(len(tiles) / cols)
    # Pad to full grid using the first tile's dimensions
    tile_h, tile_w = tiles[0].shape[:2]
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    grid = np.vstack([
        np.hstack(tiles[r * cols:(r + 1) * cols])
        for r in range(rows)
    ])
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbs", type=Path, default=_DEFAULT_VERBS)
    ap.add_argument("--reference", type=Path, default=_DEFAULT_REF)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()

    print(f"Loading verbs from {args.verbs}")
    verbs = load_verbs(args.verbs)
    print(f"  {len(verbs)} verbs loaded")

    print(f"Loading reference {args.reference}")
    img_bgr = cv2.imread(str(args.reference))
    if img_bgr is None:
        print(f"ERROR: cannot read {args.reference}", file=sys.stderr)
        return 2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print("Loading LivePortrait...")
    renderer = VerbRenderer.from_default_checkpoints()

    print(f"Rendering {len(verbs)} verbs into a {args.cols}-col grid...")
    grid = render_grid(renderer, verbs, img_rgb, cols=args.cols)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Wrote {args.out}  ({grid.shape[1]}×{grid.shape[0]} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
