# Model3 Aggregate Feature Extraction - Bug Fix

## Status Update (2026-05-31)

This approach is retained for history but is no longer the active model3 path.
Model3 now uses full holistic landmarks (`[T, 543, 3]`) instead of aggregate 3-feature vectors.

## Issue
Model3 was failing with input shape mismatch error:
```
Got invalid dimensions for input: serving_default_inputs:0
index: 1 Got: 55 Expected: 543
index: 2 Got: 1086 Expected: 3
```

## Root Cause
The model expects input shape `[batch, 543, 3]` where:
- 543 = number of frames in sequence
- 3 = aggregate features per frame

However, the application was providing `[batch, 55, 1086]` which is the raw MediaPipe Holistic keypoints format (55 keypoints × 2 coords, flattened).

## Solution Implemented

### 1. Feature Extraction Mode
Added `feature_mode: "aggregate_3d"` to the model configuration to signal that aggregate features should be extracted instead of raw keypoints.

### 2. Aggregate Feature Extraction
Implemented `_extract_aggregate_3d_features()` method in `landmarker.py` that computes 3 aggregate features from the 55-point MediaPipe Holistic landmarks:

1. **Dominant hand center X** - Horizontal position of the active hand
2. **Dominant hand center Y** - Vertical position of the active hand  
3. **Hand spread** - Average distance from center to all hand points (measure of hand openness)

The method:
- Extracts left hand (points 13-33) and right hand (points 34-54)
- Determines which hand is more active (more non-zero points)
- Computes the center of the dominant hand
- Calculates hand spread as average distance from center
- Returns `[1, 3]` array with the 3 features

### 3. Integration Points

**landmarker.py:**
- Added `feature_mode` field to `InputConfig`
- Added feature extraction call in `_process_holistic()` method
- Stores `_active_config` to access feature_mode during processing

**recognizer.py:**
- Updated `_process_sequence()` to handle both aggregate features (2D) and keypoints (3D)
- Detects aggregate mode when `stacked.ndim == 2`
- For aggregate features: creates tensor as `[batch, frames, features]`
- For keypoints: uses existing logic with `tensor_format` config

**model_config.py:**
- Added `feature_mode` field to `InputConfig` (default: "full")
- Added `tensor_format` field to `InferenceConfig` (default: "nodes_first")

**model3/model.json:**
- Set `"feature_mode": "aggregate_3d"` in input config
- Set `"tensor_format": "frames_first"` in inference config

## Files Modified
1. `python-backend/src/recognition/landmarker.py`
2. `python-backend/src/recognition/recognizer.py`
3. `python-backend/src/models/model_config.py`
4. `python-backend/models/sign/model3/model.json`

## Testing
To test the fix:
1. Start the application
2. Select Model3 from the UI
3. Verify the model loads without shape errors
4. Test real-time recognition and verify predictions are produced
5. Check that the input tensor shape is `[1, 543, 3]`

## Notes
- The 3 aggregate features are a simplified representation compared to full 55-point keypoints
- Model accuracy may vary depending on how well these 3 features capture the sign information
- If predictions are poor, consider adjusting the features (e.g., add velocity, hand size, etc.)
- The sequence length of 543 frames (~18 seconds at 30fps) is quite long - may need adjustment for practical use

## Status
✅ Bug fixed - ready for testing
