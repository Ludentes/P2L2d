# Third-Party Licenses

## LivePortrait (KwaiVGI)

- **Repository**: https://github.com/KwaiVGI/LivePortrait
- **Code license**: MIT
- **Model weights license**: MIT (KlingTeam/LivePortrait on HuggingFace)
- **Commercial use**: permitted for code and weights

## InsightFace (research-only restriction)

LivePortrait uses InsightFace's `buffalo_l` face detection models for source-image cropping.

- **Code license**: MIT
- **Pretrained models license**: **non-commercial research only**
- **Upstream statement**: LivePortrait README notes "should remove and replace InsightFace's detection models to fully comply with the MIT license" for commercial use

### Mitigation plan

For Phase 1 smoke test and research work: use InsightFace as LivePortrait ships it.

**Before any distribution**: swap InsightFace face detection for MediaPipe's `FaceDetector` (already a project dependency). LivePortrait's face cropping logic is in `src/utils/cropper.py` — replacement surface is small (face bbox + 5 keypoints for alignment).

## PowerHouseMan/ComfyUI-AdvancedLivePortrait

- **License**: MIT
- **Our use**: port of `calc_fe()` slider→δ mapping, attributed in source comments
- **Commercial use**: permitted
