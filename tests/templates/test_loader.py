"""Tests for templates.loader — template discovery and loading."""
import shutil

import pytest

from templates.loader import Template, _TEMPLATES_DIR, load_template


def test_load_template_returns_template():
    t = load_template("humanoid-anime")
    assert isinstance(t, Template)
    assert t.name == "humanoid-anime"


def test_template_has_schema():
    t = load_template("humanoid-anime")
    assert t.schema.dim == 13
    assert t.schema.names[0] == "AngleX"


def test_template_has_model_path():
    t = load_template("humanoid-anime")
    assert t.model_path.name == "model.pt"
    assert "humanoid-anime" in str(t.model_path)


def test_template_has_curves():
    t = load_template("humanoid-anime")
    assert "EyeLOpen" in t.curves.curves


def test_load_nonexistent_template_raises():
    with pytest.raises(FileNotFoundError):
        load_template("nonexistent-template")


def test_template_without_curves_gets_empty_set():
    tmp = _TEMPLATES_DIR / "_test_minimal"
    tmp.mkdir(exist_ok=True)
    try:
        shutil.copy(_TEMPLATES_DIR / "humanoid-anime" / "schema.toml", tmp / "schema.toml")
        t = load_template("_test_minimal")
        assert len(t.curves.curves) == 0
    finally:
        shutil.rmtree(tmp)
