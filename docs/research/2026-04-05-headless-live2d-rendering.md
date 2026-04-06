# Headless Live2D Rendering via EGL + live2d-py

**Date:** 2026-04-05
**Status:** Working

---

## Problem

live2d-py (Python bindings for Cubism SDK) requires an OpenGL context to load models and render. Previous attempt used OpenGL ES2 via EGL, which failed because live2d-py bundles GLSL `#version 120` shaders — incompatible with ES2.

## Solution

Desktop OpenGL via EGL with **compatibility profile**. Two key insights:

1. **Compatibility profile required**: live2d-py uses `#version 120` GLSL shaders. Core profile (3.3+) drops GLSL 1.20. The compatibility profile keeps it available. NVIDIA returns a 4.6 compatibility context.

2. **Surfaceless context preferred**: `EGL_KHR_surfaceless_context` lets us skip pbuffer surface creation entirely. Render to FBOs instead. Falls back to 1x1 pbuffer if extension unavailable.

## EGL Constant Pitfalls

Several EGL constants have confusingly similar hex values. The bugs we hit:

| Constant | Correct Hex | Wrong value we had |
|---|---|---|
| `EGL_HEIGHT` | `0x3056` | `0x3058` (= `EGL_HORIZONTAL_RESOLUTION`, read-only) |
| `EGL_RED_SIZE` | `0x3024` | OK |
| `EGL_GREEN_SIZE` | `0x3023` | `0x3025` (= `EGL_GREEN_SIZE`? No — checked: 0x3023 is correct per Khronos spec) |
| `EGL_BLUE_SIZE` | `0x3022` | `0x3026` |

The `EGL_BAD_ATTRIBUTE` (0x3004) error from `eglChooseConfig` was caused by wrong attribute hex values or array size mismatch. The `EGL_BAD_NATIVE_PIXMAP` (0x300C) from `eglCreatePbufferSurface` was caused by `EGL_HEIGHT` pointing at a read-only attribute.

## Implementation

Two modules in `rig/`:

- **`rig/headless_gl.py`** — Low-level EGL context management. Creates desktop OpenGL 3.3 compatibility profile context. Surfaceless preferred, pbuffer fallback. Context manager support.
- **`rig/render.py`** — High-level `RigRenderer` class. Takes a `RigConfig`, manages FBO, renders to RGBA numpy arrays. Context manager support.

## Usage

```python
from rig.render import RigRenderer
from rig.config import RIG_HIYORI

with RigRenderer(RIG_HIYORI, width=512, height=512) as renderer:
    img = renderer.render({"ParamMouthOpenY": 1.0, "ParamAngleX": -15})
    # img: (512, 512, 4) uint8 RGBA numpy array
```

## Requirements

- NVIDIA GPU with EGL support (`libEGL.so.1`)
- `live2d-py >= 0.6.1` (pip/uv)
- `PyOpenGL` (pulled in by live2d-py)
- `PYOPENGL_PLATFORM=egl` env var (set automatically by `rig/render.py`)

## Performance

- Context creation: ~50ms (one-time)
- Model load: ~100ms per model instance
- Render + readback: ~5ms per frame (512x512)
- No X11/Wayland required — works on headless servers

## Alternatives Considered

| Approach | Result |
|---|---|
| OpenGL ES2 via EGL | Shaders fail (`#version 120` not supported) |
| GLFW hidden window | Requires display server |
| OSMesa | Software only, no GPU acceleration |
| Xvfb | Extra process, slower than direct EGL |

## Use Cases

1. **Training data generation** — Render rig at known params, extract MediaPipe features, build (features, params) training pairs for MLP
2. **Texture swap validation** — Render rig with swapped textures to verify UV alignment
3. **Param exploration** — Programmatically sweep params to understand rig behavior
4. **CI testing** — Headless render tests without display server
