"""Template loader — loads schema, model path, and response curves from a template directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlp.curves import ResponseCurveSet
from mlp.data.live_portrait.template_schema import TemplateSchema, load_schema

_TEMPLATES_DIR = Path(__file__).parent


@dataclass
class Template:
    """A loaded template: schema + model path + response curves."""

    name: str
    schema: TemplateSchema
    model_path: Path
    curves: ResponseCurveSet


def load_template(name: str) -> Template:
    """Load a template by name from the templates directory.

    Args:
        name: Template directory name (e.g. "humanoid-anime").

    Returns:
        A Template with schema, model path, and curves loaded.

    Raises:
        FileNotFoundError: If the template directory or schema.toml is missing.
    """
    template_dir = _TEMPLATES_DIR / name
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    schema_path = template_dir / "schema.toml"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.toml not found in {template_dir}")

    schema = load_schema(schema_path)
    model_path = template_dir / "model.pt"

    curves_path = template_dir / "curves.toml"
    curves = ResponseCurveSet.from_toml(curves_path) if curves_path.exists() else ResponseCurveSet()

    return Template(name=name, schema=schema, model_path=model_path, curves=curves)
