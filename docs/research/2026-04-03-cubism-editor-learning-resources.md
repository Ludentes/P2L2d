# Research: Live2D Cubism Editor — Learning Resources and Starter Models

**Date:** 2026-04-03
**Sources:** 18 sources (live2d.com/en/learn/sample, docs.live2d.com/en/cubism-editor-tutorials, docs.live2d.com/en/cubism-editor-manual, cubism.live2d.com/sample-data/js/download.js, community.live2d.com, booth.pm, YouTube, plus direct inspection of 8 downloaded ZIP files)

---

## Executive Summary

Live2D provides free, directly downloadable sample models with `.cmo3` source files — no login required. The download page at `https://www.live2d.com/en/learn/sample/` links through JS to direct ZIP files at `https://cubism.live2d.com/sample-data/bin/`. There are roughly 30 models total. All confirmed `.cmo3` files are included, verified by direct ZIP inspection.

The official tutorial system at `docs.live2d.com/en/cubism-editor-tutorials/top/` is well-structured for beginners and covers the full creation workflow in order. Adding custom parameters is covered in the manual under "About Parameters" and "Add/Delete Keys to/from parameters". There is no login wall on any of this — all tutorial pages and documentation are openly accessible.

None of the official sample models have multi-sheet atlases on the FREE tier, but **Hiyori Momose PRO** has a 2-sheet atlas (`texture_00.png` + `texture_01.png`), and **Haru Greeter PRO** (Japanese only) has a 2-sheet atlas plus PSD files for material division learning. Both are freely downloadable despite being labelled PRO.

Community-recommended models for learning skew heavily toward the official Live2D samples. BOOTH.pm cmo3 searching is difficult (it indexes Japanese tags) and there is no curated English-language community list of third-party source-file models.

---

## Key Findings

### Official sample models with `.cmo3` source files

All models listed at `https://www.live2d.com/en/learn/sample/` include `.cmo3` files in their ZIP. Downloads are direct, no account required. Base URL: `https://cubism.live2d.com/sample-data/bin/`

#### Free-tier models (no Cubism PRO license needed to open)

| Model | Focus / Notes | ZIP URL | Size | Date |
|---|---|---|---|---|
| **Simple Model** | Absolute beginner: 3 params only (head tilt, blink, mouth). Minimal structure. `simple_t01.cmo3`. Single 1024px texture. | `/simple/simple.zip` | ~small | — |
| **Mark-kun (free)** | Beginner-tagged. Simple deformers + physics. PSD available separately. `mark_free_t04.cmo3`. Single 2048px texture. | `/mark_free/mark_free_en.zip` | 1.9 MB | 2021/06/10 |
| **Hiyori Momose (free)** | Simplified version of Hiyori. Good intro to deformer hierarchy + skinning. `hiyori_free_t08.cmo3`. Single 2048px texture. | `/hiyori_free/hiyori_free_en.zip` | 12.4 MB | 2021/06/10 |
| **Epsilon (free)** | Standard character with expression system (`exp3.json`). `Epsilon_free_t02.cmo3`. Single 2048px texture. Good for learning expression params. | `/epsilon/Epsilon_free.zip` | — | — |
| **Hibiki (free)** | Classic full-body model. Simple structure. Expression `.can3`. Single texture. | `/hibiki/hibiki.zip` | — | — |

#### PRO-labelled models (freely downloadable, but require Cubism Editor PRO to open the `.cmo3` fully)

Live2D Cubism Editor FREE edition has feature restrictions (e.g. max 100 art meshes per model). The PRO-labelled samples can still be opened read-only in the free editor, but editing may be limited. A **42-day PRO trial** is available.

| Model | Focus / Notes | ZIP URL | Size | Date |
|---|---|---|---|---|
| **Hiyori Momose (PRO)** | Full version. **2-sheet texture atlas** (`texture_00.png` + `texture_01.png`, both 2048px). Skinning, physics, pose switching. `hiyori_pro_t11.cmo3`. | `/hiyori_pro/hiyori_pro_en.zip` | 31.8 MB | 2023/03/08 |
| **Niziiro Mao (PRO)** | VTuber-oriented. **Blend Shapes**, multiply/screen color effects, wide head XY range. `mao_pro_t06.cmo3`. Single 4096px texture. | `/mao_pro/mao_pro_en.zip` | 77.2 MB | 2025/03/14 |
| **Kei (PRO + FREE version combined)** | Motion-sync / lip-sync showcase. Two variants in one ZIP: `kei_basic_free` (simpler) and `kei_vowels_pro` (A-I-U-E-O preset). Both include `.cmo3`. | `/kei/kei_en.zip` | 18.4 MB | 2023/10/17 |
| **Jin Natori (PRO)** | Sample for the making-of tutorial video. Arm changes, pose switching, expressions. `natori_pro_t06.cmo3`. Single 4096px texture. | `/natori/natori_pro_en.zip` | 24.8 MB | 2021/06/10 |
| **Ren Foster (PRO)** | Newest (Cubism 5.3, Jan 2026). Alpha blend masks, offscreen drawing for semi-transparency. `ren_t01.cmo3`. Single 4096px texture. | `/ren_pro/ren_pro_en.zip` | 44.5 MB | 2026/01/20 |
| **Rice Glassfield (PRO)** | Extension Interpolation and Inverted Masks. Sideways (fighting game style). `rice_pro.cmo3`. | `/rice_pro/rice_pro_en.zip` | 12.5 MB | 2021/06/10 |
| **Miara (PRO)** | Draw Order groups. Full-body animation. | `/miara/miara_pro_en.zip` | 38.3 MB | 2020/09/17 |
| **Parameter Controller Sample (PRO)** | IK/controller rigging. Not a character, it's an animation sample. | `/param_ctrl_pro/param_ctrl_pro_en.zip` | 14.7 MB | 2025/03/14 |
| **Hiyori (video version, PRO)** | Form Animation demo. | `/hiyori_movie/hiyori_movie_pro_en.zip` | 16.7 MB | 2023/03/08 |

#### Japanese-only models (no English ZIP, but `.cmo3` included)

- **Haru Greeter (PRO)**: `/haru/haru_greeter_pro_jp.zip` — **2-sheet texture atlas** + **2 PSD files** (material-separation PSD + import PSD). Directly relevant to garment/outfit structure learning. No English version.
- **Hatsune Miku (PRO/FREE)**: `/miku/miku_pro_jp.zip` — Skinning showcase for hair physics.
- **Haru (full version, PRO)**: `/haru/haru.zip` — Outfit changes + audio.
- **Koharu & Haruto (PRO)**: `/SDcharacter/koharu_haruto.zip` — PSD files available, SD-style.

#### Mark-kun PSD (standalone)
The source PSD for Mark-kun can be downloaded directly without installing anything:
`https://cubism.live2d.com/sample-data/bin/mark_psd/mark_free_Import.psd` (confirmed 200 OK)

---

### Official tutorial resources

All at `https://docs.live2d.com/en/cubism-editor-tutorials/top/`

#### Quick-start (under 20 min each)
1. **"Live2D by Diagrams"** (`/figure/`) — Conceptual overview using shapes. Covers: PSD import, ArtMesh, parameters, deformers, warp deformer, rotation deformer, parent-child hierarchy. PSD downloadable. **Best starting point.**
2. **"Learn about Eye Blinking in 10 minutes"** (`/eye-blink/`) — First hands-on tutorial. PSD downloadable. Explicitly recommended by Live2D as "No. 1 moment of feeling fun."
3. **"Learn about A-E-I-O-U in 20 minutes"** (`/mouth-aiueo/`) — Mouth parameter setup.

#### Basic Tutorial (6-part series, videos)
1. `/psd/` — Illustration Processing (Photoshop/CSP layer setup). Import PSD downloadable.
2. `/import/` — Preparing to Move the Illustration (mesh editing, deformers)
3. `/expression/` — Adding Facial Expressions
4. `/animator/` — Adding Body Movement
5. `/xy/` — Adding XY Facial Movement
6. (Animation creation)

Uses Hiyori Momose as the example model throughout. Completed model downloadable.

#### Template Tutorial
- `/template/` — Easy Modeling with Template Function (snap-fit rigging for standard VTuber models)

#### Parameter Controller (IK, new in 2025)
- `/controller-settings/` — Controller target settings
- `/target-tracking-settings/` — Target tracking
- `/animation-using-controller/` — Creating animations using controllers
- `/parameter-controller-tips/` — Advanced tips

#### Embedded Use Tutorial (6-part series)
For when you want to export for use in VTS/nizima LIVE etc.
- `/preparing-the-model/` through `/operation-of-viewer-unity-version/`

#### Other
- `/natori_making/` — Making-of video for Jin Natori. Covers full production process.
- `/motion-hint/` — Motion quality improvement tips
- `/function/` — Useful functions overview
- `/blendmode-offscreendrawing/` — Blend mode and offscreen drawing (Cubism 5.3)
- `/model-introduction/` — Making a model introduction video

#### Manual reference pages directly relevant to BCI parameter work
- `https://docs.live2d.com/en/cubism-editor-manual/parameter/` — About Parameters
- `https://docs.live2d.com/en/cubism-editor-manual/edit-parameters/` — **Add/Delete Keys to/from Parameters** (the page for adding custom parameters). Process: select object → select/create parameter → click "Add 3 Keyforms" → sculpt shapes at each key value.
- `https://docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/` — All standard parameter IDs (ParamAngleX/Y/Z, ParamEyeLOpen, ParamMouthForm, etc.) and their ranges
- `https://docs.live2d.com/en/cubism-editor-manual/texture-atlas-edit/` — Texture atlas editing
- `https://docs.live2d.com/en/cubism-editor-manual/workflow/` — Production flow overview
- `https://docs.live2d.com/en/cubism-editor-manual/glossary/` — Terminology glossary

---

### Community recommendations

**community.live2d.com** (the official forum, `https://community.live2d.com/`):
- Categories: Help (1.2K threads), Tips and Tricks (54 threads), Feature Requests (187)
- Tag "beginner" has 30 threads; tag "parameters" has 51 threads
- Forum thread "Running Live2D Cubism on Linux! ft. Wine & DXVK" confirmed to exist

**Reddit r/Live2D** (could not fetch wiki directly — bot block):
- The subreddit exists at `https://www.reddit.com/r/Live2D/`
- Community generally recommends the official Basic Tutorial and Hiyori as first study model (widely referenced in tutorials as the "standard" example)

**YouTube — confirmed channels/videos found:**
- Official Live2D EN channel (`@Live2D_EN`) has tutorial content
- "I Learned Live2D In 24 Hours" — community tutorial video (confirmed found in search results)
- "Live2D Tutorial For Beginners | Animation" — community tutorial
- "Live2D by diagrams" (official, subtitled) — referenced by official tutorial page
- GeekCon DX 2025 had a "Live2D Rigging for Beginners" segment (Russian-language, confirmed)

No specific third-party creator is confirmed as the dominant English community resource. The Doki Doki Drawing / Narabbit / Reiga channels are frequently referenced in the community but could not be confirmed from search results in this session.

---

### Models with advanced structure (multi-sheet atlas, garment params)

| Model | Multi-sheet? | Outfit/visibility params? | PSD included? | Notes |
|---|---|---|---|---|
| Hiyori Momose PRO | Yes: 2 sheets (2048+2048) | Pose switching via `pose3.json` | No | Best confirmed open-source example with 2-sheet atlas |
| Haru Greeter PRO | Yes: 2 sheets (2048+2048) | Arm change ("outfit change") | Yes: 2 PSDs | Japanese ZIP only; PSDs show material division before/after. Directly teaches garment layer separation. |
| Haru (full) PRO | Yes (inferred from complexity) | Arm changes + outfit changes | No | Has `pose3.json` for outfit switching |
| Natori PRO | No: 1 sheet (4096) | Arm changes, pose switching | No | `pose3.json` included |
| Mao PRO | No: 1 sheet (4096) | No visible outfit params | No | Blend Shapes, expression system |

No official sample has visibility toggles implemented as simple 0/1 boolean parameters (like HaiMeng's garment toggle). In the official samples, outfit/part visibility is managed via `pose3.json` (exclusive-group switching) rather than individual toggle parameters. However, parts visibility can be set by controlling the "Opacity" property of a Part group — this is done by binding a parameter keyform to an ArtMesh or Part's opacity.

---

## Recommendation for this project

**Start here (day 1–2):**
1. Read `https://docs.live2d.com/en/cubism-editor-tutorials/figure/` ("Live2D by Diagrams") — 10-min concept overview
2. Download Mark-kun free: `https://cubism.live2d.com/sample-data/bin/mark_free/mark_free_en.zip` (1.9 MB). Open `mark_free_t04.cmo3`. This is explicitly tagged "For beginners" and is minimal enough to navigate without being overwhelmed.
3. Follow the Basic Tutorial series (6 videos at `/psd/` through animation), using the downloadable import PSD

**For adding BCI parameters to an existing rig:**
- The relevant manual page is `https://docs.live2d.com/en/cubism-editor-manual/edit-parameters/`
- Process: Open the `.cmo3` → Parameter palette → right-click or use the "Add Parameter" button to create e.g. `ParamJawClench` (range 0–1) or `ParamFocusLevel` (range 0–1) → select the ArtMeshes/deformers that should react → click "Add 2 Keyforms" → go to value 0 (rest), then go to value 1 (clenched), sculpt deformation
- Custom parameters do not need to match the standard parameter list; any ID string works. The SDK/VTS will drive them externally.

**For understanding multi-sheet atlas structure:**
- Download Hiyori PRO: `https://cubism.live2d.com/sample-data/bin/hiyori_pro/hiyori_pro_en.zip` (31.8 MB, requires PRO editor or 42-day trial)
- It has `texture_00.png` + `texture_01.png` — open the `.cmo3` and look at the Texture Atlas palette to see how parts are distributed across sheets
- Haru Greeter JP: `https://cubism.live2d.com/sample-data/bin/haru/haru_greeter_pro_jp.zip` — the PSD files inside show the pre-separation and post-separation layer structure, which directly maps to how atlas UVs are laid out

**For VTuber-specific parameter complexity (closer to HaiMeng):**
- Mao PRO (`mao_pro_en.zip`, 77.2 MB) is the best official example of a full VTuber-grade rig with Blend Shapes, expression system, and multiply/screen color
- Requires PRO editor to open the `.cmo3`

---

## Open Questions

1. **Third-party BOOTH.pm source models**: BOOTH.pm uses Japanese tags primarily; no curated English list of "free `.cmo3` learning models" was found. The correct Japanese search term is `Live2D 素材` or `Live2D cmo3 配布`. This was not confirmed accessible in this research session.

2. **Cubism Editor FREE vs PRO restrictions**: The free editor caps at 100 art meshes per model and restricts some advanced features (blend shapes, extended interpolation). Mark-kun free and Simple Model are explicitly designed to fit within FREE tier limits. Hiyori PRO and Mao PRO almost certainly exceed 100 meshes.

3. **VTS-native parameter IDs**: VTube Studio maps to specific standard parameter IDs (e.g. `ParamMouthOpenY`, `ParamEyeLOpen`). Custom BCI parameters like `ParamJawClench` would need to be driven via VTS's parameter injection API (already researched in Plan 5). There is no restriction on adding them to a `.cmo3` — VTS will expose all parameters it finds in the `model3.json`.

4. **Community-recommended third-party tutorial creators**: Not confirmed in this session. Reddit r/Live2D is likely the best source; the wiki and pinned posts could not be scraped directly (bot protection).

---

## Sources

1. `https://www.live2d.com/en/learn/sample/` — Official sample data page (fetched 2026-04-03)
2. `https://cubism.live2d.com/sample-data/js/download.js` — Download URL manifest (fetched 2026-04-03, all URLs extracted and verified 200 OK)
3. `https://docs.live2d.com/en/cubism-editor-tutorials/top/` — Tutorial index (fetched 2026-04-03)
4. `https://docs.live2d.com/en/cubism-editor-tutorials/figure/` — Live2D by Diagrams tutorial
5. `https://docs.live2d.com/en/cubism-editor-tutorials/eye-blink/` — 10-minute blink tutorial
6. `https://docs.live2d.com/en/cubism-editor-tutorials/psd/` — Basic Tutorial step 1
7. `https://docs.live2d.com/en/cubism-editor-manual/top/` — Editor Manual index (link enumeration)
8. `https://docs.live2d.com/en/cubism-editor-manual/edit-parameters/` — Add/Delete Keys manual page
9. `https://docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/` — Standard parameter IDs
10. `https://community.live2d.com/` — Official community forum (fetched 2026-04-03)
11. `https://cubism.live2d.com/sample-data/bin/simple/simple.zip` — Direct ZIP inspection
12. `https://cubism.live2d.com/sample-data/bin/mark_free/mark_free_en.zip` — Direct ZIP inspection
13. `https://cubism.live2d.com/sample-data/bin/hiyori_free/hiyori_free_en.zip` — Direct ZIP inspection
14. `https://cubism.live2d.com/sample-data/bin/hiyori_pro/hiyori_pro_en.zip` — Direct ZIP inspection (confirmed 2-sheet atlas)
15. `https://cubism.live2d.com/sample-data/bin/epsilon/Epsilon_free.zip` — Direct ZIP inspection
16. `https://cubism.live2d.com/sample-data/bin/kei/kei_en.zip` — Direct ZIP inspection
17. `https://cubism.live2d.com/sample-data/bin/natori/natori_pro_en.zip` — Direct ZIP inspection
18. `https://cubism.live2d.com/sample-data/bin/haru/haru_greeter_pro_jp.zip` — Direct ZIP inspection (confirmed 2-sheet atlas + PSD files)
