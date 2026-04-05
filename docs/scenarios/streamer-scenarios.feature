# Streamer Reference Scenarios
#
# Real-world use cases for the portrait-to-live2d pipeline.
# Personas:
#   - Hana: indie VTuber, digital artist, provides her own artwork
#   - Marcus: IRL streamer wanting a cartoon avatar from a selfie
#   - Yuki: established VTuber wanting a second model for a sub-persona
#   - Dr. Reyes: neuro/BCI experimenter, wants EEG-driven expressions
#   - Mika: existing VTuber who already has a .moc3 and wants BCI added
#
# Out of scope for these scenarios:
#   - Commercial licensing (all outputs are academic-only per HaiMeng EULA)
#   - 3D / VRM output
#   - Voice-driven lip sync (separate system)

# ─────────────────────────────────────────────────────────────
# FEATURE 1: Core portrait-to-model pipeline
# ─────────────────────────────────────────────────────────────

Feature: Generate a Live2D model from a portrait image

  Background:
    Given the pipeline service is running
    And ComfyUI is running at http://127.0.0.1:8188
    And the HaiMeng rig template is available

  Scenario: Indie VTuber generates her first model from original artwork
    Given Hana has a 2048x2048 PNG of her anime-style character
    And the character faces forward with a neutral expression
    When she submits the portrait to the pipeline
    Then the pipeline extracts face, hair, and clothing regions
    And generates 9 texture sheets matching the artwork style
    And assembles a .model3.json using the HaiMeng rig template
    And the output passes Live2D SDK validation
    And Hana can load the model in VTube Studio without errors

  Scenario: IRL streamer generates an avatar from a selfie photo
    Given Marcus has a 1920x1080 JPEG selfie taken in good lighting
    And his face is clearly visible and centered
    When he submits the photo to the pipeline
    Then the pipeline detects a realistic photographic style
    And generates texture sheets in a semi-realistic style matching the photo
    And the face in the generated model is recognizably similar to Marcus
    And the output .model3.json loads in VTube Studio

  Scenario: Portrait is too low resolution to generate usable textures
    Given a user submits a 256x256 pixel portrait
    When the pipeline validates the input
    Then the pipeline rejects the image with a clear error message
    And suggests a minimum resolution of 512x512

  Scenario: Portrait does not contain a face
    Given a user submits an image showing only a landscape
    When the pipeline runs face detection
    Then the pipeline fails with error "No face detected in portrait"
    And no partial output files are left on disk

  Scenario: Portrait contains multiple faces
    Given a user submits a group photo with three people
    When the pipeline runs face detection
    Then the pipeline detects multiple faces
    And prompts the user to crop to a single face
    And does not proceed until a single-face crop is provided

  Scenario: Pipeline completes and delivers a clean output bundle
    Given a valid portrait has been processed successfully
    When the pipeline finalises the output
    Then the output directory contains:
      | File                        | Description                        |
      | character.model3.json       | Main model descriptor              |
      | character.moc3              | Compiled motion data               |
      | textures/texture_00.png     | Body base texture (4096×4096)      |
      | textures/texture_01.png     | Hair texture (4096×4096)           |
      | textures/texture_02.png     | Sleeve texture (4096×4096)         |
      | textures/texture_03.png     | Skirt texture (4096×4096)          |
      | textures/texture_04.png     | Trouser texture (4096×4096)        |
      | textures/texture_05.png     | Shirt body texture (4096×4096)     |
      | textures/texture_07.png     | Boot texture (4096×4096)           |
    And the .model3.json references all texture files correctly
    And the model validates without errors in the Live2D Cubism Viewer

# ─────────────────────────────────────────────────────────────
# FEATURE 2: Style diversity
# ─────────────────────────────────────────────────────────────

Feature: Output style matches the input portrait's visual style

  Background:
    Given the pipeline service is running
    And Flux Kontext is the active generation model

  Scenario: Flat anime illustration produces a flat anime model
    Given Hana provides a flat cel-shaded anime portrait with clean outlines
    When the pipeline generates texture sheets
    Then the textures use flat cel shading with minimal gradients
    And outlines match the weight and style of the input artwork
    And the visual style is consistent across all 9 texture sheets

  Scenario: Semi-realistic photo portrait produces a semi-realistic model
    Given Marcus provides a photographic selfie
    When the pipeline generates texture sheets with semi-realistic style
    Then skin textures contain natural variation and detail
    And hair textures show strand-level detail
    And the visual style is consistent across all 9 texture sheets

  Scenario: Painterly digital artwork preserves painterly style
    Given a user provides a portrait in a painterly illustration style
    When the pipeline generates texture sheets
    Then brushstroke texture and soft edge rendering are preserved
    And the output style is consistent across all 9 texture sheets

  Scenario: Streamer requests explicit style override
    Given Marcus provides a photographic selfie
    But he wants an anime-style avatar, not a semi-realistic one
    When he sets the style parameter to "flat-anime"
    Then the pipeline uses the anime style LoRA for generation
    And the output textures are in flat cel-shaded anime style
    And the face is stylistically adapted from Marcus's likeness

  Scenario: Style consistency is maintained across all texture sheets
    Given any valid portrait has been submitted
    When the pipeline generates all 9 texture sheets
    Then all textures share the same line weight
    And all textures share the same colour palette and shading style
    And no texture sheet looks like it was generated with a different model

# ─────────────────────────────────────────────────────────────
# FEATURE 3: Face tracking and expression mapping
# ─────────────────────────────────────────────────────────────

Feature: Generated model responds to live face tracking

  Background:
    Given a valid .model3.json has been generated
    And the model is loaded in VTube Studio
    And the user's webcam is active

  Scenario: Basic face tracking drives the model in real time
    Given MediaPipe landmark detection is running
    When the user turns their head left
    Then the model's ParamAngleX decreases accordingly
    When the user opens their mouth
    Then the model's ParamMouthOpenY increases
    When the user closes their left eye
    Then the model's ParamEyeLOpen decreases

  Scenario: CartoonAlive MLP maps landmarks to all rig parameters
    Given the MLP inference service is running with the trained model
    When MediaPipe detects 478 facial landmarks from the webcam feed
    Then the MLP produces all N Live2D parameter values within 16ms
    # N = 74 for Hiyori dev rig, 107 for HaiMeng production rig
    And all parameter values are within their valid range
    And the model animation is visually smooth at 60fps

  Scenario: Tracking continues when the user briefly looks away
    Given face tracking is active
    When the user looks away from the camera for less than 2 seconds
    Then the model holds its last expression
    When the user looks back at the camera
    Then tracking resumes without a visible snap or jump

  Scenario: Low-light conditions degrade gracefully
    Given the user's room is dimly lit
    When MediaPipe detects landmarks with confidence below 0.5
    Then the pipeline reduces parameter update frequency
    And outputs a warning indicator in the status overlay
    But the model does not freeze or crash

# ─────────────────────────────────────────────────────────────
# FEATURE 4: BCI signal integration
# ─────────────────────────────────────────────────────────────

Feature: Model responds to Muse 2 BCI signals via Muse VTuber Bridge

  # BCI→VTube Studio is handled entirely by the Muse VTuber Bridge
  # (zyphraexps/muse-vtuber). The bridge creates these custom VTS parameters:
  #   MuseBlink, MuseClench, MuseFocus, MuseRelaxation
  # portrait-to-live2d builds no BCI code. The rig needs deformers wired to
  # these parameters in Cubism Editor (rigging work, deferred).

  Background:
    Given a model with deformers wired to MuseClench/MuseFocus/MuseRelaxation/MuseBlink is loaded in VTube Studio
    And the Muse VTuber Bridge is running with --vts and connected to a Muse 2 headband

  Scenario: Jaw clench triggers a visible expression without opening the mouth
    Given the user's face is neutral and mouth is closed
    When the user clenches their jaw firmly
    Then MuseClench rises to 1.0 within 200ms
    And the model shows the jaw clench expression
    And ParamMouthOpenY remains unchanged
    # This is the unique value-add: EMG detects the clench invisibly

  Scenario: Rising focus level produces a focused expression
    Given the user is streaming and concentration is increasing
    When the EEG theta/beta ratio indicates high focus for 3 seconds
    Then MuseFocus rises smoothly from 0 toward 1.0
    And the model's expression reflects increased concentration
    And the transition is gradual rather than snapping

  Scenario: Relaxation signal produces a calm expression
    Given the user has been in a relaxed state for 5 seconds
    When EEG alpha power indicates relaxation
    Then MuseRelaxation rises above 0.7
    And the model's expression softens to reflect calm

  # Heartbeat (PPG) is not currently supported by the Muse VTuber Bridge.
  # Deferred until PPG output is added to the bridge.

  Scenario: Muse 2 headband disconnects mid-stream
    Given BCI signals are actively driving the model
    When the Muse 2 Bluetooth connection drops
    Then all BCI parameters hold their last value for 2 seconds
    And then interpolate smoothly to their neutral (0.0) values over 1 second
    # Handled by the Muse VTuber Bridge — not portrait-to-live2d code
    And a reconnection attempt begins automatically in the background
    And the face tracking parameters continue uninterrupted throughout

  Scenario: Muse 2 not available at stream start
    Given the Muse 2 headband is not connected
    When the user launches the model in VTube Studio
    Then the model loads and face tracking works normally
    And BCI parameters are inactive but present in VTube Studio's parameter panel
    And no error is thrown because of missing BCI input

# ─────────────────────────────────────────────────────────────
# FEATURE 5: Adding BCI parameters to an existing model
# ─────────────────────────────────────────────────────────────

Feature: Add BCI parameter support to an existing Live2D model

  # Mika already has a commissioned Live2D model and wants BCI support.
  # No file modification is needed. The Muse VTuber Bridge creates
  # MuseBlink/MuseClench/MuseFocus/MuseRelaxation as VTS custom parameters
  # automatically when run with --vts. The only rigging work is wiring
  # deformers to those parameters in Cubism Editor.

  Background:
    Given Mika has an existing .model3.json and .moc3 from a professional rigger
    And the Muse VTuber Bridge is installed and configured

  Scenario: BCI parameters appear in VTube Studio without model file changes
    When Mika launches the Muse VTuber Bridge with --vts
    Then VTube Studio shows MuseClench, MuseFocus, MuseRelaxation, MuseBlink in the custom parameters panel
    And the .model3.json and .moc3 are unchanged
    And Mika can load her model in VTube Studio without errors

  Scenario: VTube Studio receives BCI parameter values via WebSocket
    Given Mika's model is loaded in VTube Studio
    And the Muse VTuber Bridge is running with --vts
    When jaw clench is detected
    Then VTube Studio receives the MuseClench value via the VTS custom parameter API
    And the parameter is visible in VTube Studio's parameter panel
    # Even without rig wiring, the parameter value is live-injectable

  Scenario: Starting the bridge when BCI parameters already exist is idempotent
    Given the bridge has been run before and the VTS custom parameters already exist
    When the bridge is started again
    Then no duplicate parameters are created in VTube Studio
    And existing parameter values are updated normally

# ─────────────────────────────────────────────────────────────
# FEATURE 6: Garment and appearance variants
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# NOTE: Feature 6 (garment variants) is deferred.
# ParamHairToggle/ParamSkirtToggle/ParamTrouserToggle are HaiMeng-specific.
# Not implemented until HaiMeng EULA access + custom rig rigging work.
# These parameters are wired in Cubism Editor, driven by VTS hotkeys —
# not part of the MLP output or any pipeline code.
# ─────────────────────────────────────────────────────────────

Feature: Streamer can toggle clothing variants on the generated model

  Background:
    Given a model has been generated with garment toggle parameters
    And the model is loaded in VTube Studio

  Scenario Outline: Streamer toggles a garment variant
    When the streamer sets <parameter> to <value>
    Then the model displays the <result>

    Examples:
      | parameter         | value | result                          |
      | ParamHairToggle   | 1     | ponytail hairstyle              |
      | ParamHairToggle   | 0     | default hairstyle               |
      | ParamSkirtToggle  | 1     | skirt visible, trousers hidden  |
      | ParamSkirtToggle  | 0     | skirt hidden                    |
      | ParamTrouserToggle| 1     | trousers visible, skirt hidden  |

  Scenario: Yuki uses two garment presets for different stream segments
    Given Yuki has saved a "casual" preset with skirt and short hair
    And a "formal" preset with trousers and hair pinned up
    When she switches from "casual" to "formal" during a stream
    Then all garment parameters transition within one animation frame
    And no z-fighting or visibility glitches occur during the swap

# ─────────────────────────────────────────────────────────────
# FEATURE 7: Pipeline performance expectations
# ─────────────────────────────────────────────────────────────

Feature: Pipeline completes within acceptable time on the local GPU

  Background:
    Given the pipeline is running on an RTX 5090 with 32GB VRAM
    And ComfyUI is idle with no queued jobs

  Scenario: Full portrait-to-model pipeline completes in under 30 minutes
    Given Hana submits a 1024x1024 portrait
    When the pipeline runs end-to-end at full resolution
    Then texture generation completes in under 25 minutes
    And model assembly completes in under 5 minutes
    And the total wall time is under 30 minutes

  Scenario: Pipeline handles a queue of three portraits without crashing
    Given three different portraits are submitted in quick succession
    When all three are queued in ComfyUI
    Then each portrait is processed sequentially
    And all three produce valid output bundles
    And VRAM is freed between jobs
