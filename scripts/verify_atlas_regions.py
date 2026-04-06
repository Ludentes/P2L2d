"""Quick visual verification of atlas region assignments.

Renders Hiyori with target drawable groups highlighted in distinct colors
to verify left/right hair assignments and tex1 body/clothing split.

Output: /tmp/atlas_verify_<group>.png for each group
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / ".venv/lib/python3.12/site-packages"))

import numpy as np
from PIL import Image

MODEL3 = Path("/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels/hiyori_vts/hiyori.model3.json")

GROUPS = {
    "hair_side_left": (
        ["ArtMesh109","ArtMesh111","ArtMesh112","ArtMesh113","ArtMesh114",
         "ArtMesh115","ArtMesh116","ArtMesh65","ArtMesh58","ArtMesh57"],
        (1.0, 0.0, 0.0, 1.0),
    ),
    "hair_side_right": (
        ["ArtMesh66","ArtMesh117","ArtMesh118","ArtMesh119","ArtMesh120",
         "ArtMesh121","ArtMesh122","ArtMesh123","ArtMesh124","ArtMesh125",
         "ArtMesh126","ArtMesh127","ArtMesh128","ArtMesh129","ArtMesh130",
         "ArtMesh131","ArtMesh132","ArtMesh133","ArtMesh134","ArtMesh135",
         "ArtMesh136","ArtMesh137","ArtMesh59","ArtMesh60"],
        (0.0, 1.0, 0.0, 1.0),
    ),
    "body_tex1": (
        ["ArtMesh68","ArtMesh69","ArtMesh70","ArtMesh71","ArtMesh72","ArtMesh73",
         "ArtMesh74","ArtMesh75","ArtMesh76","ArtMesh77","ArtMesh78","ArtMesh79",
         "ArtMesh80","ArtMesh81","ArtMesh86","ArtMesh92",
         "ArtMesh102","ArtMesh103","ArtMesh104","ArtMesh97",
         "ArtMesh83","ArtMesh85","ArtMesh88","ArtMesh91"],
        (0.2, 0.6, 1.0, 1.0),
    ),
    "clothing_tex1": (
        ["ArtMesh99","ArtMesh100","ArtMesh101","ArtMesh93","ArtMesh94",
         "ArtMesh95","ArtMesh96","ArtMesh98","ArtMesh105","ArtMesh106","ArtMesh110",
         "ArtMesh82","ArtMesh84","ArtMesh87","ArtMesh89","ArtMesh90"],
        (1.0, 0.6, 0.0, 1.0),
    ),
}

W, H = 512, 512


def main():
    import live2d.v3 as live2d
    from OpenGL import GL
    from rig.headless_gl import create_headless_context

    ctx = create_headless_context()
    live2d.init()
    live2d.glInit()

    # Set up FBO once
    fbo = GL.glGenFramebuffers(1)
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, W, H, 0,
                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                               GL.GL_TEXTURE_2D, tex, 0)
    rbo = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rbo)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, W, H)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                  GL.GL_RENDERBUFFER, rbo)
    GL.glViewport(0, 0, W, H)

    def read_pixels() -> np.ndarray:
        pixels = GL.glReadPixels(0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        buf = bytes(pixels) if not isinstance(pixels, bytes) else pixels
        img = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)
        return np.flipud(img).copy()

    def load_model() -> live2d.LAppModel:
        m = live2d.LAppModel()
        m.LoadModelJson(str(MODEL3))
        m.Resize(W, H)
        m.SetAutoBreathEnable(False)
        m.SetAutoBlinkEnable(False)
        return m

    # Full render
    print("Rendering full model...")
    model = load_model()
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    live2d.clearBuffer(0.15, 0.15, 0.15, 1.0)
    model.Update()
    model.Draw()
    GL.glFinish()
    full = read_pixels()
    full_img = Image.fromarray(full)
    full_img.save("/tmp/atlas_verify_full.png")
    print("  saved /tmp/atlas_verify_full.png")

    # Group renders
    for group_name, (ids, color) in GROUPS.items():
        print(f"Rendering {group_name} ({len(ids)} drawables)...")
        model = load_model()
        all_ids = model.GetDrawableIds()
        target_set = set(ids)
        for idx, did in enumerate(all_ids):
            if did in target_set:
                model.SetDrawableScreenColor(idx, *color)
            else:
                model.SetDrawableMultiplyColor(idx, 0.15, 0.15, 0.15, 1.0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        live2d.clearBuffer(0.1, 0.1, 0.1, 1.0)
        model.Update()
        model.Draw()
        GL.glFinish()
        highlighted = read_pixels()

        combined = Image.new("RGBA", (W * 2 + 4, H), (40, 40, 40, 255))
        combined.paste(full_img, (0, 0))
        combined.paste(Image.fromarray(highlighted), (W + 4, 0))
        out = f"/tmp/atlas_verify_{group_name}.png"
        combined.save(out)
        print(f"  saved {out}")

    live2d.dispose()
    ctx.destroy()
    print("Done.")


if __name__ == "__main__":
    main()
