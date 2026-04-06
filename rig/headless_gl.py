"""Headless OpenGL context via EGL for programmatic Live2D rendering.

Creates a desktop OpenGL context (compatibility profile) using EGL.
Works on NVIDIA GPUs without a display server (X11/Wayland not required).

The compatibility profile is required because live2d-py uses GLSL #version 120
shaders, which are not available in core profile contexts.

Two strategies are attempted in order:
  1. Surfaceless context (EGL_KHR_surfaceless_context) -- no surface needed,
     render to FBOs only. Preferred.
  2. Pbuffer surface fallback -- creates a small offscreen pixel buffer.

Usage:
    from rig.headless_gl import create_headless_context
    with create_headless_context() as ctx:
        # ... do OpenGL / live2d-py work ...
        pass
"""
from __future__ import annotations

import ctypes
from ctypes import c_int, c_void_p

# ---------------------------------------------------------------------------
# EGL constants (hex values from EGL 1.5 spec + Khronos extension registry)
# ---------------------------------------------------------------------------
_EGL_OPENGL_API = 0x30A2
_EGL_RENDERABLE_TYPE = 0x3040
_EGL_OPENGL_BIT = 0x0008
_EGL_SURFACE_TYPE = 0x3033
_EGL_PBUFFER_BIT = 0x0001
_EGL_BLUE_SIZE = 0x3022
_EGL_GREEN_SIZE = 0x3023
_EGL_RED_SIZE = 0x3024
_EGL_NONE = 0x3038
_EGL_WIDTH = 0x3057
_EGL_HEIGHT = 0x3056
_EGL_CONTEXT_MAJOR_VERSION = 0x3098
_EGL_CONTEXT_MINOR_VERSION = 0x30FB
_EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD
_EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT = 0x00000002
_EGL_EXTENSIONS = 0x3055


class HeadlessGLContext:
    """Manages an EGL display + optional surface + desktop OpenGL context."""

    def __init__(self, display, surface, context, egl):
        self._display = display
        self._surface = surface  # may be c_void_p(0) for surfaceless
        self._context = context
        self._egl = egl

    def make_current(self):
        self._egl.eglMakeCurrent(
            self._display, self._surface, self._surface, self._context,
        )

    def destroy(self):
        egl = self._egl
        egl.eglMakeCurrent(
            self._display, c_void_p(0), c_void_p(0), c_void_p(0),
        )
        if self._surface:
            egl.eglDestroySurface(self._display, self._surface)
        egl.eglDestroyContext(self._display, self._context)
        egl.eglTerminate(self._display)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.destroy()
        return False


def _setup_egl_types(egl):
    """Set ctypes argtypes/restypes for EGL functions."""
    egl.eglGetDisplay.restype = c_void_p
    egl.eglGetDisplay.argtypes = [c_void_p]
    egl.eglInitialize.restype = c_int
    egl.eglInitialize.argtypes = [
        c_void_p, ctypes.POINTER(c_int), ctypes.POINTER(c_int),
    ]
    egl.eglChooseConfig.restype = c_int
    egl.eglChooseConfig.argtypes = [
        c_void_p, ctypes.POINTER(c_int), ctypes.POINTER(c_void_p),
        c_int, ctypes.POINTER(c_int),
    ]
    egl.eglCreatePbufferSurface.restype = c_void_p
    egl.eglCreatePbufferSurface.argtypes = [
        c_void_p, c_void_p, ctypes.POINTER(c_int),
    ]
    egl.eglBindAPI.restype = c_int
    egl.eglBindAPI.argtypes = [c_int]
    egl.eglCreateContext.restype = c_void_p
    egl.eglCreateContext.argtypes = [
        c_void_p, c_void_p, c_void_p, ctypes.POINTER(c_int),
    ]
    egl.eglMakeCurrent.restype = c_int
    egl.eglMakeCurrent.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
    egl.eglGetError.restype = c_int
    egl.eglGetError.argtypes = []
    egl.eglQueryString.restype = ctypes.c_char_p
    egl.eglQueryString.argtypes = [c_void_p, c_int]
    egl.eglDestroySurface.restype = c_int
    egl.eglDestroySurface.argtypes = [c_void_p, c_void_p]
    egl.eglDestroyContext.restype = c_int
    egl.eglDestroyContext.argtypes = [c_void_p, c_void_p]
    egl.eglTerminate.restype = c_int
    egl.eglTerminate.argtypes = [c_void_p]


def create_headless_context(
    width: int = 1,
    height: int = 1,
) -> HeadlessGLContext:
    """Create a headless desktop OpenGL context via EGL.

    Uses compatibility profile so that GLSL #version 120 shaders (used by
    live2d-py) work correctly. Prefers surfaceless context when available,
    falls back to a 1x1 pbuffer otherwise.

    Args:
        width: Pbuffer width (only used if surfaceless is unavailable).
        height: Pbuffer height (only used if surfaceless is unavailable).

    Returns:
        A HeadlessGLContext. The context is already current on return.
    """
    egl = ctypes.cdll.LoadLibrary("libEGL.so.1")
    _setup_egl_types(egl)

    # --- Display + init ---
    display = egl.eglGetDisplay(c_void_p(0))
    if not display:
        raise RuntimeError("eglGetDisplay failed")

    major, minor = c_int(), c_int()
    if not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
        raise RuntimeError(f"eglInitialize failed: 0x{egl.eglGetError():04x}")

    # --- Bind desktop OpenGL (not ES) ---
    if not egl.eglBindAPI(_EGL_OPENGL_API):
        raise RuntimeError(f"eglBindAPI(EGL_OPENGL_API) failed: 0x{egl.eglGetError():04x}")

    # --- Check for surfaceless extension ---
    exts_raw = egl.eglQueryString(display, _EGL_EXTENSIONS)
    exts = exts_raw.decode() if exts_raw else ""
    has_surfaceless = "EGL_KHR_surfaceless_context" in exts

    # --- Choose config ---
    # Request OpenGL + pbuffer support (pbuffer bit needed for config
    # selection even if we ultimately use surfaceless).
    config_attribs = (c_int * 11)(
        _EGL_SURFACE_TYPE, _EGL_PBUFFER_BIT,
        _EGL_BLUE_SIZE, 8,
        _EGL_GREEN_SIZE, 8,
        _EGL_RED_SIZE, 8,
        _EGL_RENDERABLE_TYPE, _EGL_OPENGL_BIT,
        _EGL_NONE,
    )
    config = c_void_p()
    num_configs = c_int()
    if not egl.eglChooseConfig(
        display, config_attribs, ctypes.byref(config), 1, ctypes.byref(num_configs),
    ):
        raise RuntimeError(f"eglChooseConfig failed: 0x{egl.eglGetError():04x}")
    if num_configs.value == 0:
        raise RuntimeError("No EGL config found for desktop OpenGL + pbuffer")

    # --- Create context (compatibility profile for #version 120 support) ---
    ctx_attribs = (c_int * 7)(
        _EGL_CONTEXT_MAJOR_VERSION, 3,
        _EGL_CONTEXT_MINOR_VERSION, 3,
        _EGL_CONTEXT_OPENGL_PROFILE_MASK,
        _EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
        _EGL_NONE,
    )
    context = egl.eglCreateContext(display, config, c_void_p(0), ctx_attribs)
    if not context:
        # Fall back to default (no version/profile request)
        context = egl.eglCreateContext(display, config, c_void_p(0), None)
    if not context:
        raise RuntimeError(f"eglCreateContext failed: 0x{egl.eglGetError():04x}")

    # --- Make current: surfaceless preferred, pbuffer fallback ---
    surface = c_void_p(0)

    if has_surfaceless:
        ok = egl.eglMakeCurrent(display, c_void_p(0), c_void_p(0), context)
        if ok:
            return HeadlessGLContext(display, surface, context, egl)

    # Pbuffer fallback
    surf_attribs = (c_int * 5)(
        _EGL_WIDTH, width, _EGL_HEIGHT, height, _EGL_NONE,
    )
    surface = egl.eglCreatePbufferSurface(display, config, surf_attribs)
    if not surface:
        raise RuntimeError(
            f"eglCreatePbufferSurface failed: 0x{egl.eglGetError():04x}"
        )

    if not egl.eglMakeCurrent(display, surface, surface, context):
        raise RuntimeError(f"eglMakeCurrent failed: 0x{egl.eglGetError():04x}")

    return HeadlessGLContext(display, surface, context, egl)
