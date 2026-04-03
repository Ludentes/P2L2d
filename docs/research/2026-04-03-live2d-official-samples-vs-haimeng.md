# Live2D Official Sample Models vs. HaiMeng — Technical Comparison

**Date:** 2026-04-03
**Purpose:** Evaluate whether Live2D's official SDK sample models could replace HaiMeng as the base rig in a Textoon-style generative pipeline.

---

## 1. Source of Truth

All data below was extracted directly from:

- `github.com/Live2D/CubismWebSamples` (develop branch) — `Samples/Resources/`
- `github.com/Human3DAIGC/Textoon` (main branch) — `assets/model_configuration.json`, `live2d-chatbot-demo/`
- Live2D license page: `live2d.com/en/learn/sample/` and `live2d-sample-model-terms_en.html`

No secondary sources were used for the structural data.

---

## 2. Official Sample Models — Inventory

Eight models ship in both CubismWebSamples and CubismNativeSamples. All are identical across both repos.

| Model | Texture atlas dir | Sheets | Sheet size | Parameters | Parts | Body coverage |
|-------|-------------------|--------|------------|------------|-------|---------------|
| **Haru** | `Haru.2048/` | 2 | 2048×2048 each | 42 | 20 | Full body, standing |
| **Hiyori** | `Hiyori.2048/` | 2 | 2048×2048 each | 70 | 26 | Full body, standing |
| **Mao** | `Mao.2048/` | 1 | 2048×2048 | 132 | 33 | Full body, standing |
| **Mark** | `Mark.2048/` | 1 | 2048×2048 | 21 | 17 | Full body, standing |
| **Natori** | `Natori.2048/` | 1 | 2048×2048 | 96 | 32 | Full body, standing |
| **Ren** | `Ren.2048/` | 1 | 2048×2048 | 73 | 52 | Full body, standing |
| **Rice** | `Rice.2048/` | 2 | 2048×2048 each | 96 | 30 | Full body, standing |
| **Wanko** | `Wanko.1024/` | 1 | 1024×1024 | 25 | 32 | Chibi/non-human |

### Texture total area comparison

All official sample models use **2048×2048 px sheets**. Maximum total texture area per model:

- 2-sheet models (Haru, Hiyori, Rice): 2 × 2048² = 8.4 M pixels
- 1-sheet models (Mao, Natori, Ren): 1 × 2048² = 4.2 M pixels

### Files included in the SDK repos

Each model directory contains:
```
ModelName.moc3          — compiled rig (binary, not editable)
ModelName.model3.json   — runtime descriptor (textures, motions, expressions)
ModelName.cdi3.json     — parameter/part display names
ModelName.physics3.json — physics simulation config
ModelName.pose3.json    — pose groups (most models)
expressions/            — .exp3.json expression files (most models)
motions/                — .motion3.json keyframe animations
```

**What is NOT included:**
- `.cmo3` — Cubism Editor project file (the rigging source). **Not present anywhere in either SDK repo.**
- `.can3` — animation source files. **Not present.**
- `.psd` — illustration source layers. **Not present.**

The `.cmo3` and `.psd` source files are available **only via the separate per-model download** on `live2d.com/en/learn/sample/` (search confirms the download page offers them). They are not part of the SDK package that ships on GitHub.

---

## 3. HaiMeng Dataset — Structure Summary

From `assets/model_configuration.json` and the `live2d-chatbot-demo/` demo model:

### Runtime files

```
female_01Arkit_6.moc3
female_01Arkit_6.model3.json
female_01Arkit_6.cdi3.json
female_01Arkit_6.physics3.json
female_01Arkit_6.4096/
    texture_00.png  (4096×4096, ~4.1 MB — body base, thighs, calves, eyes)
    texture_01.png  (4096×4096, ~1.8 MB — all hair variants)
    texture_02.png  (4096×4096, ~1.8 MB — sleeves)
    texture_03.png  (4096×4096, ~2.3 MB — skirts)
    texture_04.png  (4096×4096, ~0.5 MB — trousers)
    texture_05.png  (4096×4096, ~0.6 MB — shirt body)
    texture_06.png  (4096×4096, ~0.7 MB — unused/extra)
    texture_07.png  (4096×4096, ~0.3 MB — boots)
    texture_08.png  (4096×4096, ~0.9 MB — breast variant shirts)
config.json         — Textoon-specific ARKit mapping + setparameter dict
```

### Parameters

107 total (cdi3.json). Mix of:
- Standard Live2D face/body params: `ParamAngleX/Y/Z`, `ParamEyeLOpen`, `ParamBrowLY`, `ParamBodyAngleX/Y/Z`, etc.
- **Full ARKit face blendshapes** (24 params): `ParamJawOpen`, `ParamMouthClose`, `ParamMouthFunnel`, `ParamMouthPucker`, `ParamMouthStretchLeft/Right`, `MouthSmileLeft`, `MouthFrownLeft/Right`, `MouthShrugUpper/Lower`, `MouthUpperUpLeft/Right`, `ParamMouthLowerDownLeft/Right`, `ParamMouthRollUpper/Lower`, `ParamMouthPressLeft/Right`, `MouthCheekPuff`, `ParamMouthDimpleLeft/Right`
- Custom Textoon config params (named `Param`, `Param47`, `Param48`, `Param54`, `Param57`, `Param59`, `Param60`): these are boolean toggles controlling rig visibility (ponytail on/off, skirt on/off, trousers on/off, breast size variant)
- ~50 internal body/physics params named `Param2`–`Param53` etc. (private rig internals)

### Parts (24 parts, Chinese names)

Lower body (`Part8`), upper body (`Part7`), left/right hands, face, left/right eyes, mouth, nose, front/back hair (with variant switching), clothing zones.

### The Textoon pipeline's relationship to the atlas

`model_configuration.json` encodes **explicit pixel-coordinate UV crop tables** for every swappable clothing piece across all 9 texture sheets. For example:

```json
"skirt":     {"x": 72,  "y": 61, "w": 1268, "h": 2554, "name": "texture_03"}
"trousers":  {"x": 82,  "y": 46, "w": 871,  "h": 2520, "name": "texture_04"}
"left_boot": {"x": 506, "y": 63, "w": 284,  "h": 1326, "name": "texture_07"}
```

The pipeline renders a generated image to the working canvas (3360×5040 px from PSD), then uses these coordinates to extract each clothing region and splat it onto the corresponding 4096×4096 texture sheet at the known UV destination. This is what `utils/transfer_part_texture.py:extract_part_to_texture()` does.

---

## 4. Side-by-Side Comparison

| Dimension | HaiMeng | Best Official Sample (Hiyori/Natori/Mao) |
|-----------|---------|------------------------------------------|
| **Format** | Live2D Cubism 4, `.moc3` | Live2D Cubism 4/5, `.moc3` |
| **Texture sheets** | **9 × 4096×4096** | 1–2 × 2048×2048 |
| **Total texture area** | **150 M pixels** | 4.2–8.4 M pixels (**18–36× less**) |
| **Body coverage** | Full body, standing, full legs | Full body, standing (most models) |
| **Clothing variants in atlas** | 9 dedicated sheets: hair, sleeves, shirts (2 variants), skirts, trousers, boots, extra | None — single fixed outfit baked into atlas |
| **Swappable clothing parts** | Yes — each garment type occupies its own texture sheet or a known UV region | No — clothing is part of the single baked atlas, not swappable |
| **ARKit blendshapes** | **Yes — 24 ARKit mouth/jaw parameters** natively rigged | None — all sample models use generic `ParamMouthOpenY` + mouth form (no per-blendshape mouth rigging) |
| **Parameter count** | 107 | 21 (Mark) – 132 (Mao, but many are special effects) |
| **Rig config parameters** | 7 boolean toggles for garment/hair/body type switching | None |
| **Source files (.cmo3, .psd)** | Not publicly available (proprietary to Textoon/HaiMeng creators) | Available on official download page (separate from SDK repo) |
| **Parameter naming** | Mix of standard + internal (`Param2`–`Param53`) | Standard `ParamAngle*`, `ParamEye*`, etc. |
| **License** | EULA.pdf in Textoon repo — restrictive, Textoon-specific | "Free Material License" — General/Small-Scale commercial use OK; Medium/Large entities: private testing only |

---

## 5. Compatibility Assessment

### 5.1 Texture atlas structure

**Not compatible.** The entire Textoon pipeline is built around the assumption of a 9-sheet × 4096 atlas where each clothing type occupies a dedicated sheet or a hardcoded UV region. Official sample models have 1–2 sheets at 2048 px, with all clothing baked into the single atlas as a fixed outfit. There is no separate "sleeve sheet" or "skirt sheet" to swap.

To adapt, you would need to:
1. Re-rig an official model with a new 9-sheet atlas structure (requires Cubism Editor + `.cmo3` source).
2. Manually define new UV coordinate tables in `model_configuration.json` for the new atlas.

This is not a drop-in replacement — it would require rebuilding the rig's texture assignment from scratch.

### 5.2 Body coverage

**Compatible in principle.** Hiyori, Natori, Mao, and Ren are all full-body standing characters comparable to HaiMeng's silhouette. The Textoon pipeline doesn't require a specific pose, only that the mesh covers the full body.

### 5.3 ARKit blendshapes

**Not compatible.** Zero official sample models have ARKit-style per-blendshape mouth parameters. They all use the standard 2-parameter mouth (`ParamMouthForm` + `ParamMouthOpenY`). HaiMeng has 24 ARKit mouth blendshapes rigged. To use an official sample model with the ARKit `config.json` mapping, the rig would need to be rebuilt in Cubism Editor to add those deformer chains — which again requires `.cmo3` source.

### 5.4 Source file availability (.cmo3)

**The `.cmo3` files are available via the Live2D sample download page** (not in the GitHub SDK repos, but distributed separately). This means it is technically possible to open the official sample models in Cubism Editor and modify the rig — add texture sheets, add ARKit parameters, etc.

However, this is substantial rig engineering work (weeks, not hours) and would produce a derivative of a Live2D character subject to the Free Material License terms.

### 5.5 License implications

The Free Material License allows General Users and Small-Scale Enterprises (annual revenue under ¥10M) to use, modify, and distribute the models commercially. Medium and Large entities may only use them for private testing. Redistribution of derivatives must comply with per-character terms.

HaiMeng's EULA (in the Textoon repo) is a separate, more restrictive agreement specific to the Textoon project.

---

## 6. Verdict

**Official sample models cannot replace HaiMeng in the current Textoon pipeline without significant re-engineering.** The structural differences are fundamental:

1. **Texture architecture mismatch** — 9 × 4096 px specialized sheets vs. 1–2 × 2048 px baked atlases. The UV coordinate extraction/injection system in `transfer_part_texture.py` would need complete redefinition.

2. **No ARKit rig** — Adding 24 ARKit mouth blendshapes requires re-rigging in Cubism Editor. The `.cmo3` source is available from Live2D's download page, making this *possible* but not trivial.

3. **No garment-switching architecture** — Official samples have no boolean rig parameters for hiding/showing clothing variants. HaiMeng's `Param47`/`Param48`/`Param54`/`Param60` etc. are bespoke rig features.

**If starting a new pipeline from scratch**, Hiyori or Natori would be the strongest candidates (70–96 parameters, full body, 26–32 parts, `.cmo3` available) as a foundation to build upon, provided you're willing to:
- Restructure the texture atlas to 4–9 sheets
- Add ARKit deformers
- Add visibility-toggle parameters for clothing variants

This is a multi-week Cubism Editor project, not a drop-in swap.

---

## 7. Reference Data

**CubismWebSamples repo:** `github.com/Live2D/CubismWebSamples` (branch: `develop`)  
**Textoon repo:** `github.com/Human3DAIGC/Textoon`  
**Live2D sample download page:** `live2d.com/en/learn/sample/`  
**Free Material License:** `live2d.com/eula/live2d-free-material-license-agreement_en.html`  
**Sample model terms:** `live2d.com/eula/live2d-sample-model-terms_en.html`

Sources:
- [Live2D Sample Data (for Free)](https://www.live2d.com/en/learn/sample/)
- [Terms of Use for Live2D Cubism Sample Data](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)
- [Free Material License Agreement](https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html)
- [Can I use the sample models for commercial purposes?](https://help.live2d.com/en/other/other_16/)
- [Textoon GitHub — Official Repo](https://github.com/Human3DAIGC/Textoon)
- [Textoon paper (arXiv)](https://arxiv.org/html/2501.10020v1)
