"""Main entrypoint — starts face tracking bridge and/or OBS bridge.

Usage:
  # Face tracking only (webcam → MLP → VTS)
  uv run python -m runtime.runner --vts

  # OBS ambient effects only (EEG backend → OBS Color Correction)
  uv run python -m runtime.runner --obs

  # Both together
  uv run python -m runtime.runner --vts --obs

  # Benchmark MLP inference speed
  uv run python -m runtime.runner --benchmark

Prerequisites:
  VTS:  VTube Studio running with hiyori model loaded, API enabled (port 8001)
  OBS:  OBS running, obs-websocket enabled (port 4455), EEG_Overlay source +
        EEG_Ambient Color Correction filter configured
  EEG:  ZyphraExps backend running (python -m backend.main --synthetic)
        for OBS bridge
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from mlp.infer import load_predictor
from rig.config import RIG_HIYORI
from runtime.obs_bridge import ObsBridge
from runtime.param_bridge import run as run_param_bridge


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


async def _main(args: argparse.Namespace) -> None:
    tasks = []

    if args.vts:
        predictor = load_predictor(RIG_HIYORI)
        logging.getLogger(__name__).info(
            "MLP loaded — %d params, inference %.1fms",
            RIG_HIYORI.param_count,
            predictor.benchmark(n=50),
        )
        tasks.append(
            run_param_bridge(
                rig=RIG_HIYORI,
                predictor=predictor,
                camera_index=args.camera,
            )
        )

    if args.obs:
        bridge = ObsBridge()
        tasks.append(bridge.run())

    if not tasks:
        print("Nothing to run — pass --vts and/or --obs")
        return

    await asyncio.gather(*tasks)


def main() -> None:
    _setup_logging()
    p = argparse.ArgumentParser(description="portrait-to-live2d runtime")
    p.add_argument("--vts", action="store_true", help="Face tracking → VTS bridge")
    p.add_argument("--obs", action="store_true", help="EEG → OBS ambient effects")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--benchmark", action="store_true", help="Print MLP speed and exit")
    args = p.parse_args()

    if args.benchmark:
        predictor = load_predictor(RIG_HIYORI)
        ms = predictor.benchmark(n=200)
        print(f"MLP inference: {ms:.2f}ms mean over 200 runs")
        return

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
