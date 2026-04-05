"""Verb library loader — reads a verbs.toml file into VerbEntry objects."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .verb_sliders import VerbSliders


@dataclass
class VerbEntry:
    """One verb: LivePortrait slider settings + template param targets."""
    name: str
    description: str
    sliders: VerbSliders
    params: dict[str, float]  # template param name → value (MLP label)


def load_verbs(toml_path: Path) -> list[VerbEntry]:
    """Load a verbs.toml file.

    Expected structure:
        [verbs.<name>]
        description = "..."
        sliders = { blink = -15.0, ... }
        params  = { EyeLOpen = 0.0, ... }
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    verbs_table = data.get("verbs", {})
    entries: list[VerbEntry] = []
    for name, v in verbs_table.items():
        sliders_dict = v.get("sliders") or {}
        # Only pass known slider fields — tomllib returns a plain dict
        known = {f for f in VerbSliders.__dataclass_fields__}
        unknown = set(sliders_dict) - known
        if unknown:
            raise ValueError(f"verb {name!r}: unknown slider keys {unknown}")
        entries.append(
            VerbEntry(
                name=name,
                description=v.get("description", ""),
                sliders=VerbSliders(**sliders_dict),
                params=dict(v.get("params") or {}),
            )
        )
    return entries
