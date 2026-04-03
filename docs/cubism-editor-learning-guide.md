# Live2D Cubism Editor — Learning Guide for BCI Integration

**Date:** 2026-04-03  
**Goal:** Go from zero rigging experience → add custom BCI parameters to a Live2D rig.  
**Context:** Learning step before working with the HaiMeng dataset. Test the full pipeline (Cubism → VTS → Muse bridge) while waiting for HaiMeng EULA access.

---

## What You're Learning

Live2D Cubism Editor is the tool for creating and editing `.cmo3` rig source files. You need it to:
- Add custom parameters (`ParamJawClench`, `ParamFocusLevel`, `ParamRelaxation`) to a rig
- Understand the texture atlas structure (how garment layers map to UV regions)
- Export `.moc3` + `.model3.json` for use in VTube Studio

Cubism Editor runs on Windows and Mac only. On Linux, use Wine or a Windows VM.

---

## Step 1 — Install Cubism Editor

Download from `https://www.live2d.com/en/cubism/download/editor/`

Two tiers:
- **FREE**: limited to 100 art meshes per model. Fine for Mark-kun and basic learning.
- **PRO**: no mesh limit, required for Hiyori PRO and most production models. **42-day free trial available** — activate it before working with Hiyori.

---

## Step 2 — Concepts First (10 minutes)

Read [Live2D by Diagrams](https://docs.live2d.com/en/cubism-editor-tutorials/figure/) before touching any software.

It covers the mental model you need:
- **ArtMesh**: triangulated polygon layer over the illustration. This is what deforms.
- **Warp Deformer**: a cage that bends/stretches child meshes uniformly.
- **Rotation Deformer**: pivots child meshes around a point.
- **Parameter**: a named number (e.g. `ParamAngleX`, range -30 to +30). Moving it interpolates between recorded keyforms.
- **Keyform**: a snapshot of mesh/deformer positions at a specific parameter value.
- **Parent-child hierarchy**: deformers nest inside each other. Moving a parent moves all children.

---

## Step 3 — First Hands-On (Mark-kun FREE)

Download (1.9 MB, no account needed):
```
https://cubism.live2d.com/sample-data/bin/mark_free/mark_free_en.zip
```

Open `mark_free_t04.cmo3`. This model is explicitly designed for beginners — minimal mesh count, simple deformer hierarchy, fits within the FREE editor limits.

Do the [10-minute blink tutorial](https://docs.live2d.com/en/cubism-editor-tutorials/eye-blink/) using the downloadable PSD. This covers: importing art, creating a mesh, adding a parameter, recording keyforms at 0 (open) and 1 (closed).

**That single exercise is the core loop for all rigging work.** Everything else is variations on: select mesh → bind parameter → record keyforms at each value → sculpt.

---

## Step 4 — Main Learning Model (Hiyori PRO)

Hiyori Momose PRO is the community standard for learning. The official 6-part Basic Tutorial uses her throughout.

Download (31.8 MB):
```
https://cubism.live2d.com/sample-data/bin/hiyori_pro/hiyori_pro_en.zip
```

Activate the 42-day PRO trial before opening — Hiyori PRO exceeds the FREE tier mesh limit.

Work through the [Basic Tutorial series](https://docs.live2d.com/en/cubism-editor-tutorials/psd/):
1. Illustration Processing (PSD layer setup)
2. Preparing to Move the Illustration (meshes + deformers)
3. Adding Facial Expressions
4. Adding Body Movement
5. Adding XY Facial Movement
6. Animation creation

**Why Hiyori specifically for this project:** She has a 2-sheet texture atlas (`texture_00.png` + `texture_01.png`). Open the Texture Atlas palette in Cubism to see how parts are distributed across sheets — this is the same architecture HaiMeng uses (at larger scale with 9 sheets).

---

## Step 5 — Add BCI Parameters

Once you're comfortable navigating a `.cmo3`, add the Muse bridge parameters to Hiyori. This is both a learning exercise and a functional test.

Manual reference: `https://docs.live2d.com/en/cubism-editor-manual/edit-parameters/`

### The process (same for every custom parameter)

1. Open `hiyori_pro_t11.cmo3`
2. In the **Parameter** palette, click **"Add Parameter"** (+ button or right-click menu)
3. Name it: use any ID string. Suggested IDs:

| Parameter ID | Range | What it drives |
|---|---|---|
| `ParamJawClench` | 0–1 | EMG jaw clench → intensity expression (eyebrow furrow, eye squint) |
| `ParamFocusLevel` | 0–1 | EEG concentration → subtle brow raise or eye narrowing |
| `ParamRelaxation` | 0–1 | EEG alpha → drooped eyelids, softened expression |
| `ParamHeartbeat` | 0–1 | PPG pulse → optional subtle breathing sway |

4. Select the ArtMeshes or deformers that should react (e.g. eyebrow mesh for `ParamJawClench`)
5. Click **"Add 2 Keyforms"** — creates snapshots at value 0 and value 1
6. Navigate to value 0 → this is the rest pose (do nothing, it's already correct)
7. Navigate to value 1 → sculpt the mesh into the "active" shape (furrow the brow, lower the eyelid, etc.)
8. Repeat for each parameter

### Export

File → Export Runtime Files → select output folder. This produces:
- `hiyori_pro.moc3` — compiled rig
- `hiyori_pro.model3.json` — runtime descriptor (lists your new parameters)

Copy the folder to VTS `Live2DModels/` and reload. In VTS, open Model Settings — your `ParamJawClench` etc. will appear in the parameter list. Wire them to Muse VTuber Bridge custom parameter injection.

### Verify end-to-end

With Muse connected and the bridge running, the parameters should move in VTS in real time. See `docs/muse-vtuber-vts-setup-guide.md` for the full VTS WebSocket auth and parameter binding setup.

---

## Reference: Understanding Garment Visibility (for HaiMeng prep)

HaiMeng uses boolean parameters to toggle clothing visibility. Official samples use `pose3.json` (exclusive-group switching) instead, but the underlying mechanism is the same: binding a parameter to the **Opacity** property of a Part group.

To understand it: open Hiyori PRO, find a Part that gets hidden (e.g. body parts that hide during certain poses), inspect its opacity keyforms. In HaiMeng, `Param47` (ponytail toggle) works the same way — at value 0 the ponytail Part is opacity 0, at value 1 it's opacity 1.

For the Haru Greeter JP model (shows garment layer separation most clearly):
```
https://cubism.live2d.com/sample-data/bin/haru/haru_greeter_pro_jp.zip
```
Contains 2 PSD files: pre-separation and post-separation layer structure. Even without reading Japanese, the PSDs show exactly how garment components are split into separate layers before atlas packing.

---

## Standard Parameter IDs Reference

If a parameter uses a standard ID, VTS auto-maps it. Custom BCI params use non-standard IDs and are driven via the WebSocket API.

Full list: `https://docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/`

| ID | Controls | Range |
|----|----------|-------|
| `ParamAngleX` | Head yaw | -30 to 30 |
| `ParamAngleY` | Head pitch | -30 to 30 |
| `ParamAngleZ` | Head roll | -30 to 30 |
| `ParamEyeLOpen` | Left eye open | 0 to 1 |
| `ParamEyeROpen` | Right eye open | 0 to 1 |
| `ParamEyeBallX` | Eye gaze X | -1 to 1 |
| `ParamEyeBallY` | Eye gaze Y | -1 to 1 |
| `ParamBrowLY` | Left brow height | -1 to 1 |
| `ParamBrowRY` | Right brow height | -1 to 1 |
| `ParamMouthOpenY` | Mouth open | 0 to 1 |
| `ParamMouthForm` | Smile/frown | -1 to 1 |
| `ParamBodyAngleX` | Body sway X | -10 to 10 |

---

## Files Referenced

| File | What it is |
|------|-----------|
| `mark_free_t04.cmo3` | Beginner model source (inside mark_free_en.zip) |
| `hiyori_pro_t11.cmo3` | Main learning model source (inside hiyori_pro_en.zip) |
| `docs/muse-vtuber-vts-setup-guide.md` | VTS WebSocket auth + parameter binding |
| `docs/ai-live2d-pipeline-overview.md` | Full pipeline context and HaiMeng access path |
| `docs/research/2026-04-03-cubism-editor-learning-resources.md` | Full research notes with all download URLs |
