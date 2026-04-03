# Research: Real-Time Portrait-to-2D-Character Animation for VTubing

**Date:** 2026-04-03  
**Sources:** 18 sources (see below)  
**Scope:** Real-time face-driven 2D character animation from a single portrait. Excludes pre-rendered talking-head video, 3D/VRM approaches, and body animation. Focus: what works now, what's usable, what's the best bootstrap path.

---

## Executive Summary

Two fundamentally different approaches exist. **Neural warp methods** (FasterLivePortrait, MobilePortrait) take any portrait image and deform it in real-time based on webcam input — no rigging, works immediately, but the output is a warped version of the source photo rather than a stylized anime character. **Parametric Live2D methods** (CartoonAlive, Textoon) generate an actual rigged Live2D model that can be driven by face tracking at VTubing quality — but CartoonAlive's code is unreleased and Textoon is template-limited to predefined component shapes. The most important finding is that **Textoon** [1][2] — the predecessor to CartoonAlive by the same team — has fully released code (Apache 2.0) that generates working `.model3.json` Live2D files and is the best bootstrap point for building a production CartoonAlive. For immediate real-time use without any rigging, FasterLivePortrait [3] runs at 30+ FPS on an RTX 3090 and has MIT-licensed code and weights today.

---

## Key Findings

### The two paradigms and why the distinction matters for VTubing

The neural warp approach (First Order Motion Model lineage, LivePortrait, MobilePortrait) works by estimating a dense optical flow field between a source portrait and a driving face, then warping the source image accordingly. The output is the source *photo* in motion — if you upload an anime illustration, you get that illustration warping. If you upload a photo, you get a warped photo. This is fast, requires no preprocessing, and the code is available. The limitation is that it can never produce the clean, stable, cel-shaded Live2D aesthetic that traditional VTubing delivers — the warp artifacts on hair, clothing edges, and extreme expressions are visible, and the animation is video-like rather than puppet-like.

The parametric Live2D approach works by constructing an actual rigged model — a set of mesh deformers and parameter curves — and then mapping face tracking output to those parameters at runtime. This produces stable, stylized, controllable animation that composites cleanly against green screens or transparent backgrounds. Traditional VTubers use this approach. The challenge is that it normally requires hours of manual rigging work, which CartoonAlive and Textoon attempt to automate.

For VTubing specifically, the parametric approach is almost always preferable if you want the traditional streamer aesthetic. Neural warp is practical if you want a quick demo, a novelty "stream as your photo" setup, or you are willing to accept the photorealistic-warp look.

### Neural warp: what's available and production-ready

**FasterLivePortrait** [3] is the most immediately usable option. It is an optimized reimplementation of the LivePortrait [4] system (KwaiVGI, July 2024) using ONNX and TensorRT. Running on an RTX 3090 with TensorRT enabled, it achieves 30+ FPS including pre- and post-processing — not just model inference. Real-time webcam mode is one flag: `python run.py --src_image portrait.jpg --dri_video 0 --cfg configs/trt_infer.yaml --realtime`. Weights are on HuggingFace (`warmshao/FasterLivePortrait`). License is MIT. The gap: there is no virtual camera output; the animated frames appear in a display window or are saved to file. Piping to OBS requires a virtual camera wrapper (e.g., `v4l2loopback` on Linux + `ffmpeg` piping).

**MobilePortrait** [5] (CVPR 2025) achieves over 100 FPS on iPhone 14 Pro and 50+ FPS on older mobile, requiring only 16 GFLOPs versus 200–629 GFLOPs for comparable approaches. It uses a mixed explicit/implicit keypoint architecture with lightweight U-Net backbones. This is primarily a research paper — code and weight availability are not confirmed from sources reviewed. Its relevance is architectural: the 16 GFLOP budget shows that real-time portrait warping is now a solved problem even on constrained hardware.

**Thin-Plate Spline Motion Model** [6] (CVPR 2022) is older but has an ONNX version available for Video-to-Image reenactment and a Windows packaged release. It handles anime illustration sources reasonably well (unlike some newer methods that are trained predominantly on photo faces). Worth knowing as a fallback that is battle-tested.

**PersonaLive** [7] is a LivePortrait-based pipeline that adds a local Gradio UI, a REST API, and an MCP server on top of the animation engine, making it scriptable from Claude Code or other LLM agents. It requires 12GB+ VRAM. The MCP integration makes it directly usable as a "generate animation" tool in an agentic workflow.

**Viggle LIVE** [8] is a commercial SaaS offering real-time webcam character animation. It reports a 1–2 second delay between motion and output — this latency makes it unsuitable for VTubing where sub-100ms feedback is expected.

### Live2D parametric: Textoon is the key discovery

**Textoon** [1][2] was published January 2025 by the same group (Human3DAIGC, authors Chao He and Jianqiang Ren) who later built CartoonAlive. It generates Live2D characters from text descriptions and — critically — **the code is fully released under Apache 2.0** with working instructions.

The Textoon pipeline generates actual `.model3.json` Live2D model files. The architecture uses: a fine-tuned Qwen2.5-1.5B model for text parsing (>90% accuracy on component extraction), SDXL with ControlNet for appearance generation within predefined component areas, SAM2 for segmentation, and inpainting for occluded regions (back hair behind head, etc.). The rendering uses the Live2D Web SDK via Node.js. Face tracking at runtime uses MediaPipe.

The template-based limitation is real: Textoon supports predefined component shapes — 5 types of back hair, 3 mid hair, 3 front hair, 5 tops, 6 sleeves, 5 pants, 5 skirts, 6 shoes. Within those templates the appearance is generated by SDXL, so visual variety is high, but the silhouette / rig structure is constrained to the template set.

**Why this matters for CartoonAlive bootstrapping:** CartoonAlive's task is "portrait → Live2D model," which is a superset of Textoon's "text → Live2D model." Textoon already solves the hard parts: generating component textures, filling occluded regions, and assembling a `.model3.json`. The additional step CartoonAlive adds is appearance *extraction* from a portrait (instead of appearance *generation* from text), plus the animation parameter estimation MLP. Textoon's codebase is the right foundation.

### CartoonAlive's architecture in context

CartoonAlive's MLP component — 4-layer network mapping MediaPipe facial landmarks to Live2D parameter values, trained on 100,000 synthetic renders generated via PyGame + Cubism SDK — is the piece that Textoon doesn't have. Textoon's generated models use MediaPipe for runtime face tracking, but the parameter mapping is presumably hardcoded to the template rig rather than learned. CartoonAlive's contribution is a learned mapping that generalizes across face shapes.

The two projects together describe a complete system: Textoon's generation pipeline for creating the model, CartoonAlive's MLP for driving it. The fact that both are by the same team, with Textoon's code released and CartoonAlive's code pending, strongly suggests CartoonAlive will eventually release — and that Textoon's internals already contain most of the building blocks.

### Inochi2D: the open-source Live2D alternative

**Inochi2D** [9][10] is a BSD 2-clause licensed open-source framework for real-time 2D puppet animation. It includes Inochi Creator (rigging tool, replaces Cubism Editor) and Inochi Session (live performance tool with face tracking, replaces VTube Studio). Protocol support includes VMC, OpenSeeFace, and VTube Studio. The rendering runtime is available as a library.

Inochi2D's advantage for building an auto-rigging system: being fully open source under a permissive license, it is possible to write code that generates Inochi2D model files directly without depending on the proprietary Cubism SDK. The file format is documented. This sidesteps the Cubism license constraint (free for non-commercial, licensed fee for commercial use) that would affect any system that generates `.moc3` files at scale.

The disadvantage: the ecosystem is smaller, fewer models exist, the rigging tooling is less mature, and there is no equivalent to CartoonAlive targeting the Inochi2D format.

### Live2D SDK programmatic control

**live2d-py** [11] provides Python bindings to the Cubism Native SDK via a C Extension. It can load, render, and drive parameters on existing `.moc3` models — which makes it suitable for the *inference side* of a CartoonAlive-style system (running the generated model in Python, controlled by MediaPipe). It cannot create or modify model files. Rendering targets include PyGame and PyQt5.

The Cubism SDK itself can be driven headlessly via PyGame for batch rendering — this is how CartoonAlive generates its 100,000 training renders. The SDK is free for non-commercial use; commercial use requires a paid license.

---

## Comparison

| Project | Type | Real-time? | Live2D output | Code/weights | License | Bootstrap value |
|---------|------|-----------|--------------|--------------|---------|----------------|
| **FasterLivePortrait** | Neural warp | ✓ 30+ FPS RTX3090 | No (photo warp) | Yes | MIT | Use now for quick demo |
| **MobilePortrait** | Neural warp | ✓ 100+ FPS mobile | No (photo warp) | Unclear | CVPR 2025 | Architecture reference |
| **TPSMM** (CVPR 2022) | Neural warp | Partial (ONNX) | No (photo warp) | Yes | Apache 2.0 | Works on anime illustrations |
| **PersonaLive** | Neural warp (LivePortrait) | ✓ with 12GB VRAM | No | Yes | — | MCP integration for scripting |
| **Textoon** | Parametric L2D | ✓ (MediaPipe) | ✓ .model3.json | **Yes** | Apache 2.0 | **Primary bootstrap target** |
| **CartoonAlive** | Parametric L2D | ✓ 30s init | ✓ Live2D model | Pending | Apache 2.0 | MLP architecture, implement from paper |
| **Inochi2D** | Parametric (open format) | ✓ | ✓ (open format) | Yes | BSD 2-clause | Alternative to Cubism SDK dependency |
| **Viggle LIVE** | Neural warp | 1–2s delay | No | No (SaaS) | Commercial | Not viable for VTubing |

---

## Bootstrap Path

Given the goal of building a real-time portrait-to-Live2D system:

**Phase 1 — Working immediately:** Use FasterLivePortrait for a real-time demo. Upload a portrait (photo or illustration), webcam drives the animation. Not Live2D, but ships today with zero new code.

**Phase 2 — Textoon foundation:** Clone `Human3DAIGC/Textoon`. This already generates `.model3.json` from text. Replace the text → component generation with portrait → component extraction: face crop → appearance extraction (SDXL img2img or direct texture crop), hair segmentation (anime-seg), clothing segmentation (HumanParser node). The result is a Textoon-equivalent driven by portrait input instead of text.

**Phase 3 — CartoonAlive MLP:** Implement the 4-layer MLP from the paper. Generate training data by randomly sampling Live2D parameter sets, rendering via PyGame + Cubism SDK (or Inochi2D if avoiding Cubism licensing), recording MediaPipe landmark ground truth. Train until convergence. This gives you learned face-to-parameter mapping that generalizes across face shapes.

**Phase 4 — Real-time loop:** MediaPipe (already used in both Textoon and CartoonAlive) → MLP → Live2D parameter injection → live2d-py or Inochi2D runtime → virtual camera out.

The Cubism licensing question is real: if this becomes a commercial product, Inochi2D as the runtime avoids the per-product fee. If it stays non-commercial, live2d-py + Cubism SDK is the simpler path since the Textoon codebase already uses Cubism.

---

## Open Questions

- CartoonAlive code release timeline is unknown. The team has a pattern of releasing code (Textoon released, Make-A-Character HuggingFace Space exists) so release is likely but unscheduled.
- MobilePortrait weights are unconfirmed available. Worth checking the CVPR 2025 proceedings for a code link.
- Textoon's template set (5+3+3 hair types, etc.) limits the output silhouette variety. Whether this is a hard architectural constraint or a training data constraint is not specified in the paper.
- Inochi2D's file format for programmatic generation has not been verified as sufficient for auto-rigging from code — needs hands-on investigation.

---

## Sources

[1] Human3DAIGC. "Textoon: Official Repo". GitHub. https://github.com/Human3DAIGC/Textoon (Retrieved: 2026-04-03)

[2] He, Chao et al. "Textoon: Generating Vivid 2D Cartoon Characters from Text Descriptions". arXiv:2501.10020. https://arxiv.org/html/2501.10020v1 (Retrieved: 2026-04-03)

[3] warmshao. "FasterLivePortrait". GitHub. https://github.com/warmshao/FasterLivePortrait (Retrieved: 2026-04-03)

[4] KwaiVGI. "LivePortrait: Efficient Portrait Animation". https://liveportrait.github.io/ (Retrieved: 2026-04-03)

[5] Jiang et al. "MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices". CVPR 2025. arXiv:2407.05712. https://arxiv.org/html/2407.05712v1 (Retrieved: 2026-04-03)

[6] yoyo-nb. "Thin-Plate-Spline-Motion-Model". CVPR 2022. GitHub. https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model (Retrieved: 2026-04-03)

[7] neosun100. "PersonaLive". GitHub. https://github.com/neosun100/PersonaLive (Retrieved: 2026-04-03)

[8] Viggle AI. "Viggle LIVE: Transform Into Anyone Live". https://viggle.ai/viggle-live (Retrieved: 2026-04-03)

[9] Inochi2D. "Official Website". https://inochi2d.com/ (Retrieved: 2026-04-03)

[10] Various. "Inochi2D vs Live2D: A New Era in 2D Modeling". Oreate AI Blog. https://www.oreateai.com/blog/inochi2d-vs-live2d-a-new-era-in-2d-modeling/eb509e20740a161468b996f073a142cd (Retrieved: 2026-04-03)

[11] EasyLive2D. "live2d-py README". GitHub. https://github.com/EasyLive2D/live2d-py/blob/main/README.en.md (Retrieved: 2026-04-03)

[12] He, Chao et al. "CartoonAlive: Towards Expressive Live2D Modeling from Single Portraits". arXiv:2507.17327. https://arxiv.org/html/2507.17327v1 (Retrieved: 2026-04-03)

[13] emilianavt. "Best VTuber Software". GitHub Gist. https://gist.github.com/emilianavt/cbf4d6de6f7fb01a42d4cce922795794 (Retrieved: 2026-04-03)

[14] instant-high. "Thin-plate-spline-motion-model-ONNX-Reenactment". GitHub. https://github.com/instant-high/Thin-plate-spline-motion-model-ONNX-Reenactment (Retrieved: 2026-04-03)

[15] Inochi2D/inochi-creator. "GitHub Repository". https://github.com/Inochi2D/inochi-creator (Retrieved: 2026-04-03)
