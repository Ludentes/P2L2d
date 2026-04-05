"""Programmatic LivePortrait-based verb rendering for MLP training data.

Uses the vendored LivePortrait pipeline (third_party/LivePortrait) to apply
parametric expression sliders to a source portrait and produce variations.
No ComfyUI runtime needed.
"""
from .verb_sliders import VerbSliders, apply_sliders
from .renderer import VerbRenderer, SourceState

__all__ = ["VerbSliders", "apply_sliders", "VerbRenderer", "SourceState"]
