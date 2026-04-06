"""Headless Live2D rig rendering via live2d-py + EGL.

Renders a Live2D model to RGBA numpy arrays without any display server.
Requires NVIDIA GPU with EGL support.

Usage:
    from rig.render import RigRenderer
    from rig.config import RIG_HIYORI

    with RigRenderer(RIG_HIYORI) as renderer:
        # Render with specific params
        img = renderer.render({"ParamMouthOpenY": 1.0, "ParamAngleX": -15})
        # img is (H, W, 4) uint8 RGBA numpy array
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

# Must be set before any OpenGL import
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from .config import RigConfig
from .headless_gl import HeadlessGLContext, create_headless_context

if TYPE_CHECKING:
    import types


class RigRenderer:
    """Renders a Live2D model headlessly with configurable parameters."""

    def __init__(self, config: RigConfig, width: int = 512, height: int = 512) -> None:
        self._config = config
        self._width = width
        self._height = height
        self._ctx: HeadlessGLContext | None = None
        self._fbo: Any = None
        self._live2d: types.ModuleType | None = None
        self._GL: types.ModuleType | None = None

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, *_exc):
        self.dispose()
        return False

    def _setup(self) -> None:
        self._ctx = create_headless_context()

        import live2d.v3 as live2d
        from OpenGL import GL

        self._live2d = live2d
        self._GL = GL

        live2d.init()
        live2d.glInit()

        # Create FBO
        W, H = self._width, self._height
        self._fbo = GL.glGenFramebuffers(1)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, W, H, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0,
        )
        rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, W, H)
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rbo,
        )
        GL.glViewport(0, 0, W, H)

    def render(self, params: dict[str, float] | None = None) -> np.ndarray:
        """Render the model with given params, return (H, W, 4) uint8 RGBA array."""
        assert self._live2d is not None and self._GL is not None, "call _setup() first"
        live2d = self._live2d
        GL = self._GL
        W, H = self._width, self._height

        model = live2d.LAppModel()
        model.LoadModelJson(str(self._config.model3_json_path))
        model.Resize(W, H)
        model.SetAutoBreathEnable(False)
        model.SetAutoBlinkEnable(False)

        if params:
            for param_id, value in params.items():
                model.SetParameterValue(param_id, value, 1.0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
        model.Update()
        model.Draw()
        GL.glFinish()

        pixels = GL.glReadPixels(0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(H, W, 4)
        return np.flipud(img).copy()

    def dispose(self) -> None:
        if self._live2d is not None:
            self._live2d.dispose()
            self._live2d = None
        if self._ctx is not None:
            self._ctx.destroy()
            self._ctx = None
