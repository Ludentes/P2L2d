# Research: ComfyUI → Live2D Art Pipeline

**Date:** 2026-04-03  
**Sources:** 15 sources (see below)

---

## Executive Summary

A practical pipeline from ComfyUI to Cubism-ready layered art exists and is usable today. The critical tool is **ComfyUI-See-through** [1], a ComfyUI plugin wrapping the academic See-through system (SIGGRAPH 2026, conditionally accepted) [2] that decomposes a single anime illustration into 19 semantic RGBA layers with inpainted occluded regions and PSD export. Used after character generation with models like NoobAI-XL or Illustrious XL, it eliminates the most tedious manual steps (segmentation + drawing-what's-hidden) while producing a PSD that can be imported directly into Live2D Cubism Editor. Expect 2–6 hours of manual cleanup before the PSD is production-rig-ready. For a completely automated end-to-end approach (portrait → Live2D model, no rigging), CartoonAlive [3] is an emerging research pipeline that produces animatable Live2D characters in under 30 seconds, though it is not yet a public tool.

---

## Key Findings

### Step 1: Generate the character art in ComfyUI

The generation side of the pipeline uses standard ComfyUI with anime-specialized checkpoints. **NoobAI-XL** and **Illustrious XL** are the current leading models for flat cell-shaded anime output [4][5]. NoobAI-XL is trained on Danbooru/e621 tags with reportedly over 4 billion training steps and produces output most consistent with standard VTuber art style. Illustrious XL is better for more detailed, higher-fidelity illustration styles.

The single most important generation constraint for Live2D prep is **art style**: flat cell shading with clean vector-like outlines segments cleanly. Painterly styles, heavy gradients, and soft-edge rendering produce layer boundaries that segmentation tools struggle with. Prompt for: plain white or isolated background, clear silhouette, simple separable accessories, no overlapping complex elements like detailed glasses in front of hair.

Resolution should be 1024×1024 or higher to give the decomposition model enough pixels to work with at the part level.

### Step 2: Decompose into layers — See-through is the primary tool

**ComfyUI-See-through** [1] is the most directly relevant tool in the current ecosystem. It wraps the See-through research system [2], which was specifically designed for Live2D workflows and trained using data bootstrapped from commercial Live2D models.

The system produces **19 semantic RGBA layers**: front hair, back hair, neck, clothing, handwear, legwear, footwear, accessories, face, eyes (left/right split), eyebrows (left/right), eyelashes, ears (left/right split), nose, mouth, eyewear. Each layer is fully inpainted — the hidden regions beneath each part (the back of the head under the hair, the body under the clothes) are "hallucinated" by the model so they exist as complete images that can be moved in the rig without revealing holes. The researchers report outputs are "largely production-ready and require only minor manual modifications" [2], supported by quantitative metrics showing significant improvement over baselines (LPIPS: 0.1549 vs. 0.2880 for comparative methods).

PSD export is built in and runs client-side via `ag-psd` without Python dependencies. Processing runs at approximately 74 seconds per 1024×1024 image. VRAM requirement is 8GB minimum, 12GB recommended.

### Limitations of See-through for production rigging

Three gaps require manual work after See-through runs:

**Layer granularity.** The 19 layers See-through produces are semantically correct but not granular enough for a professional rig. A rigger expects to work with ~50–100 layers: irises separate from pupils, upper and lower eyelashes as distinct layers, individual hair tufts (forelock, side tufts, ahoge), clothing broken into multiple garment pieces, individual fingers. After importing the See-through PSD, you'll need to open it in Photoshop or Krita and further subdivide these groups before handing off to Cubism.

**Inpainting artifacts.** The paper notes that "minor overlaps occasionally occur between layers outside body regions," and the LaMa inpainter used for hidden regions "can contain minor mistakes that typically lie in the far-back stratum" [2]. In practice this means the back-of-head layer and the body-under-clothes layer may have AI-generated patches that look slightly wrong. These require manual touch-up — usually 20–60 minutes per model depending on complexity.

**Layer naming.** The output layers use the See-through semantic names, not Live2D standard parameter IDs or conventional rigger naming. Before importing to Cubism, rename layers to match the rigger's expectations or your own naming convention.

The authors are explicit that "automating one step in isolation cannot replicate" a professional rig created holistically [2], but the tool usefully eliminates the slowest steps.

### Alternative decomposition tools

**ComfyUI-LayerDivider** [6] is a simpler decomposition tool that uses either color-based segmentation or Facebook's Segment Anything (SAM) to divide an image into PSD layers. It outputs 3–5 layers per region (base, highlights, shadows, etc.) — useful for stylistic compositing but not semantic character-part separation. Not suitable as the primary tool for Live2D prep.

**Qwen-Image-Layered** [7] is a generative model (released December 2025, Apache 2.0) that decomposes images into RGBA layers with independent editability. Its key differentiator is that it can also *regenerate* content — replace an element, recolor, resize — not just separate. It supports recursive decomposition (further splitting individual layers). Native ComfyUI support exists. However, it is a general-purpose tool not specialized for anime characters or Live2D; the decomposition boundaries follow the model's general-scene understanding rather than Live2D-relevant anatomy. VRAM requirement is 12GB+.

**ComfyUI Anime Segmentation** [8] and the **Human Parser node** [9] produce masks for body regions and clothing items, but output masks only — not inpainted layers. These work as building blocks in a custom pipeline (generate mask → inpaint the masked region to reconstruct what's hidden → save each result as a layer) but require more workflow construction than See-through.

### The end-to-end alternative: CartoonAlive

**CartoonAlive** [3] (July 2025) takes a different approach entirely: rather than decompose existing art, it generates a complete animatable Live2D model from a single portrait image in under 30 seconds. It produces layered 2D components (face, eyes, mouth, hair) bound to deformable meshes, supports 52 ARKit expression controls, and produces output compatible with existing Live2D animation controllers — no manual rigging step. The system was trained on 100,000 synthetic rendered images with an MLP mapping facial landmarks to Live2D parameters.

Limitations: ear matching is poor (landmark detection failure), pupil/iris positioning is imprecise, fine hair strands isolate inaccurately [3]. More importantly, as of April 2026 this is a research paper with a GitHub repository but not a ComfyUI plugin or polished public tool. It represents where the field is heading but is not production-usable today.

### The full practical pipeline today

```
ComfyUI (NoobAI-XL or Illustrious XL)
    → flat cell-shaded character, 1024px+, plain background
    ↓
ComfyUI-See-through
    → 19 RGBA layers + PSD (~74s, 8–12GB VRAM)
    ↓
Photoshop / Krita (manual)
    → fix inpainting artifacts, split layers further, rename to rigger convention
    → 2–6 hours
    ↓
Live2D Cubism Editor
    → import PSD, mesh, deformers, parameters, physics
    → export .moc3 + .model3.json
    ↓
VTube Studio
```

---

## Comparison

| Tool | Approach | Layers | PSD out | Live2D-specific | VRAM | Effort after |
|------|----------|--------|---------|----------------|------|-------------|
| **ComfyUI-See-through** | Diffusion decomposition | 19 semantic RGBA | Yes | Yes (trained on L2D data) | 8–12GB | 2–6h cleanup |
| **Qwen-Image-Layered** | Generative decomposition | Variable | No native | No | 12GB+ | Higher (no anime specialization) |
| **ComfyUI-LayerDivider** | SAM / color segmentation | N per region | Yes | No | Low | Not suitable solo |
| **Anime Seg + inpaint** | Mask → inpaint pipeline | DIY | Via ComfyUI-Layers | No | Moderate | High (custom pipeline) |
| **CartoonAlive** | Portrait → full L2D model | Full model | N/A (generates model) | Full | Unknown | Research only, not public tool |

---

## Open Questions

- ComfyUI-See-through's GitHub repo (jtydhr88) is a wrapper; the underlying See-through model (shitagaki-lab/see-through) is conditionally accepted to SIGGRAPH 2026 but model weights may not be fully released yet — verify before planning a production pipeline on it.
- CartoonAlive's code/weights availability is unclear from public sources.
- How well See-through handles AI-generated art (vs. human-drawn anime) is untested in sources reviewed — AI art tends to have more texture noise which may affect segmentation quality.

---

## Sources

[1] jtydhr88. "ComfyUI-See-through". GitHub. https://github.com/jtydhr88/ComfyUI-See-through (Retrieved: 2026-04-03)

[2] shitagaki-lab. "See-through: Single-image Layer Decomposition for Anime Characters". arXiv:2602.03749 (SIGGRAPH 2026, conditionally accepted). https://arxiv.org/html/2602.03749v1 (Retrieved: 2026-04-03)

[3] Unknown authors. "CartoonAlive: Towards Expressive Live2D Modeling from Single Portraits". arXiv:2507.17327. https://arxiv.org/html/2507.17327v1 (Retrieved: 2026-04-03)

[4] Various. "Illustrious vs. NoobAI-XL: Navigating the Cutting Edge of Anime AI Art". Oreate AI Blog. https://www.oreateai.com/blog/illustrious-vs-noobaixl-navigating-the-cutting-edge-of-anime-ai-art/230efde972f4e5af49e75fbb12505ce9 (Retrieved: 2026-04-03)

[5] regiellis. "ComfyUI-EasyNoobai". GitHub. https://github.com/regiellis/ComfyUI-EasyNoobai (Retrieved: 2026-04-03)

[6] jtydhr88. "ComfyUI-LayerDivider". GitHub. https://github.com/jtydhr88/ComfyUI-LayerDivider (Retrieved: 2026-04-03)

[7] ComfyUI Wiki. "Qwen-Image-Layered Released". https://comfyui-wiki.com/en/news/2025-12-19-qwen-image-layered-release (Retrieved: 2026-04-03)

[8] LyazS. "comfyui-anime-seg". GitHub. https://github.com/LyazS/comfyui-anime-seg (Retrieved: 2026-04-03)

[9] cozymantis. "human-parser-comfyui-node". GitHub. https://github.com/cozymantis/human-parser-comfyui-node (Retrieved: 2026-04-03)

[10] alessandrozonta. "ComfyUI-Layers". GitHub. https://github.com/alessandrozonta/ComfyUI-Layers (Retrieved: 2026-04-03)

[11] chflame163. "ComfyUI_LayerStyle". GitHub. https://github.com/chflame163/ComfyUI_LayerStyle (Retrieved: 2026-04-03)

[12] themoonlight.io. "Literature Review: See-through". https://www.themoonlight.io/en/review/see-through-single-image-layer-decomposition-for-anime-characters (Retrieved: 2026-04-03)

[13] KomikoAI. "How to Instantly Split Anime Art into Layers for Animation and Rigging". https://komiko.app/blog/how-to-instantly-split-anime-art-into-layers-for-animation-and-rigging (Retrieved: 2026-04-03)

[14] mintigo.store. "How to AI Rig a Model for VTubing: A Personal Guide". https://mintigo.store/how-to-ai-rig-a-model-for-vtubing-a-personal-guide/ (Retrieved: 2026-04-03)

[15] comfyui-wiki.com. "Qwen-Image-Layered Released — Image Generation Model with Layer-Based Editing Support". https://comfyui-wiki.com/en/news/2025-12-19-qwen-image-layered-release (Retrieved: 2026-04-03)
