"""Programmatic LivePortrait-based verb rendering for MLP training data.

Uses the vendored LivePortrait pipeline (third_party/LivePortrait) to apply
parametric expression sliders to a source portrait and produce variations.
No ComfyUI runtime needed.
"""
from .verb_sliders import VerbSliders, apply_sliders

__all__ = ["VerbSliders", "apply_sliders"]


def __getattr__(name: str):
    """Lazy-load renderer to avoid importing LivePortrait unless needed."""
    if name in ("VerbRenderer", "SourceState"):
        from .renderer import VerbRenderer, SourceState
        return {"VerbRenderer": VerbRenderer, "SourceState": SourceState}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
