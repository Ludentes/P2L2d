"""Detailed atlas verification — renders individual drawable groups for analysis.

Outputs to /tmp/atlas_detail_<group>.png
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL3 = Path("/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels/hiyori_vts/hiyori.model3.json")

W, H = 512, 512

GROUPS = {
    # All hair regions together to check coverage
    "hair_all": (
        ["ArtMesh50","ArtMesh56","ArtMesh63",           # hair_front
         "ArtMesh64",                                    # hair_back
         "ArtMesh109","ArtMesh111","ArtMesh112","ArtMesh113","ArtMesh114",
         "ArtMesh115","ArtMesh116","ArtMesh65","ArtMesh58","ArtMesh57",  # hair_side_left
         "ArtMesh66","ArtMesh117","ArtMesh118","ArtMesh119","ArtMesh120",
         "ArtMesh121","ArtMesh122","ArtMesh123","ArtMesh124","ArtMesh125",
         "ArtMesh126","ArtMesh127","ArtMesh128","ArtMesh129","ArtMesh130",
         "ArtMesh131","ArtMesh132","ArtMesh133","ArtMesh134","ArtMesh135",
         "ArtMesh136","ArtMesh137","ArtMesh59","ArtMesh60"],             # hair_side_right
        (1.0, 0.8, 0.0, 1.0),   # yellow — all hair
    ),
    # Unassigned drawables — what are these?
    "unassigned": (
        ["ArtMesh26","ArtMesh27","ArtMesh29","ArtMesh34",
         "ArtMesh41","ArtMesh42","ArtMesh52","ArtMesh53"],
        (1.0, 0.0, 1.0, 1.0),   # magenta
    ),
    # Body order-800 (arm detail/overlays — classified as body)
    "body_order800": (
        ["ArtMesh83","ArtMesh85","ArtMesh88","ArtMesh91"],
        (0.2, 0.6, 1.0, 1.0),   # blue
    ),
    # Clothing order-800 (clothing overlays — classified as clothing)
    "clothing_order800": (
        ["ArtMesh82","ArtMesh84","ArtMesh87","ArtMesh89","ArtMesh90"],
        (1.0, 0.6, 0.0, 1.0),   # orange
    ),
    # Order-500 right-side classified as body
    "body_order500": (
        ["ArtMesh86","ArtMesh92","ArtMesh97","ArtMesh102","ArtMesh103","ArtMesh104"],
        (0.0, 0.8, 0.8, 1.0),   # cyan
    ),
    # Order-500 left-side classified as clothing
    "clothing_order500": (
        ["ArtMesh93","ArtMesh94","ArtMesh95","ArtMesh96","ArtMesh98",
         "ArtMesh99","ArtMesh100","ArtMesh101","ArtMesh105","ArtMesh106","ArtMesh110"],
        (1.0, 0.3, 0.3, 1.0),   # red-orange
    ),
    # Background body layers (order 300-400)
    "body_background": (
        ["ArtMesh68","ArtMesh69","ArtMesh70","ArtMesh71","ArtMesh72","ArtMesh73",
         "ArtMesh74","ArtMesh75","ArtMesh76","ArtMesh77","ArtMesh78","ArtMesh79",
         "ArtMesh80","ArtMesh81"],
        (0.4, 0.4, 1.0, 1.0),   # light blue
    ),
}


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
    full_img.save("/tmp/atlas_detail_full.png")

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
        highlighted = Image.fromarray(read_pixels())

        combined = Image.new("RGBA", (W * 2 + 4, H), (40, 40, 40, 255))
        combined.paste(full_img, (0, 0))
        combined.paste(highlighted, (W + 4, 0))
        combined.save(f"/tmp/atlas_detail_{group_name}.png")
        print(f"  saved /tmp/atlas_detail_{group_name}.png")

    live2d.dispose()
    ctx.destroy()
    print("Done.")


if __name__ == "__main__":
    main()
