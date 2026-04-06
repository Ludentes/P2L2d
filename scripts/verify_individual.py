"""Render individual drawables to identify them.

Uses multiply-alpha=0 to fully hide non-targets, making targets clearly visible.
Outputs to /tmp/atlas_indiv_<id>.png
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

MODEL3 = Path("/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels/hiyori_vts/hiyori.model3.json")

W, H = 512, 512

# Drawables to isolate
TARGETS = [
    "ArtMesh102", "ArtMesh103", "ArtMesh104",  # classified body — likely socks?
    "ArtMesh97",                                 # classified body — unclear
    "ArtMesh86", "ArtMesh92",                   # arm/neck?
    "ArtMesh52", "ArtMesh53",                   # unassigned, tex=0
]


def main():
    import live2d.v3 as live2d
    from OpenGL import GL
    from rig.headless_gl import create_headless_context

    ctx = create_headless_context()
    live2d.init()
    live2d.glInit()

    fbo = GL.glGenFramebuffers(1)
    tex_gl = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_gl)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, W, H, 0,
                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                               GL.GL_TEXTURE_2D, tex_gl, 0)
    rbo = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rbo)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, W, H)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                  GL.GL_RENDERBUFFER, rbo)
    GL.glViewport(0, 0, W, H)

    def read_pixels():
        pixels = GL.glReadPixels(0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        buf = bytes(pixels) if not isinstance(pixels, bytes) else pixels
        img = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)
        return np.flipud(img).copy()

    def load_model():
        m = live2d.LAppModel()
        m.LoadModelJson(str(MODEL3))
        m.Resize(W, H)
        m.SetAutoBreathEnable(False)
        m.SetAutoBlinkEnable(False)
        return m

    # Render each target individually
    for target_id in TARGETS:
        print(f"Isolating {target_id}...")
        model = load_model()
        all_ids = model.GetDrawableIds()
        for idx, did in enumerate(all_ids):
            if did == target_id:
                # Keep target at normal color + slight brightness boost
                model.SetDrawableScreenColor(idx, 0.3, 0.3, 0.3, 1.0)
            else:
                # Hide everything else (alpha multiply = 0)
                model.SetDrawableMultiplyColor(idx, 0.0, 0.0, 0.0, 0.0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        live2d.clearBuffer(0.0, 0.0, 0.0, 1.0)
        model.Update()
        model.Draw()
        GL.glFinish()
        isolated = Image.fromarray(read_pixels())
        isolated.save(f"/tmp/atlas_indiv_{target_id}.png")
        print(f"  saved /tmp/atlas_indiv_{target_id}.png")

    live2d.dispose()
    ctx.destroy()
    print("Done.")


if __name__ == "__main__":
    main()
