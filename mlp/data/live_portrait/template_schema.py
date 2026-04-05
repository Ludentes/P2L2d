"""Template parameter schema loader (schema.toml → ordered ParamSpec list)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ParamSpec:
    name: str
    range: tuple[float, float]
    default: float
    curve: str


@dataclass
class TemplateSchema:
    params: list[ParamSpec]  # ordered — position is the MLP label index

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def dim(self) -> int:
        return len(self.params)

    def default_label(self) -> np.ndarray:
        """Default label vector (all params at their default values)."""
        return np.array([p.default for p in self.params], dtype=np.float32)

    def index_of(self, name: str) -> int:
        for i, p in enumerate(self.params):
            if p.name == name:
                return i
        raise KeyError(f"param {name!r} not in template schema")

    def apply_verb_params(
        self, label: np.ndarray, verb_params: dict[str, float]
    ) -> np.ndarray:
        """Return a copy of label with verb_params overlaid."""
        out = label.copy()
        for name, value in verb_params.items():
            idx = self.index_of(name)
            out[idx] = value
        return out


def load_schema(toml_path: Path) -> TemplateSchema:
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    entries = data.get("params", [])
    params = [
        ParamSpec(
            name=e["name"],
            range=tuple(e["range"]),
            default=float(e["default"]),
            curve=e.get("curve", "linear"),
        )
        for e in entries
    ]
    return TemplateSchema(params=params)
