# Model2 OpenPose Integration Notes

## Summary

Model2 (WLASL Pose-TGCN) expects OpenPose keypoints. The previous prediction issues were caused by MediaPipe Holistic keypoints feeding an OpenPose-trained model. This is resolved by switching model2 to `landmark_source: "openpose"`.

## Required Setup

Install OpenPose and set the environment variable:

```
OPENPOSE_DIR=C:\path\to\openpose
```

The folder must contain:
- `python/` (OpenPose Python bindings)
- `models/` (OpenPose model files)
- `bin/` (Windows DLLs)

If your models are in a different location, set `input.openpose_model_folder` in `model.json`.

## Performance Notes

- OpenPose is heavier than MediaPipe and may require a GPU for real-time performance.
- Use `openpose_net_resolution` in `model.json` to trade speed for accuracy.

## Failure Behavior

If OpenPose is not installed or `OPENPOSE_DIR` is not set, model2 will fail to initialize. In that case, switch to model1 or model3 until OpenPose is available.

---

**Date**: 2026-06-02
**Status**: OpenPose required for model2
