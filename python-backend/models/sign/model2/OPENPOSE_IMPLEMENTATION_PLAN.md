# OpenPose Implementation Plan for Model2

## Goal
Implement OpenPose keypoint extraction specifically for Model2 to make it work correctly, while keeping MediaPipe for Model1.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Recognition Pipeline                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Camera     │→ │  Landmarker  │→ │  Recognizer  │      │
│  │   Capture    │  │              │  │              │      │
│  └──────────────┘  └──────┬───────┘  └──────────────┘      │
│                            │                                 │
│                     ┌──────┴──────┐                         │
│                     │             │                         │
│              ┌──────▼─────┐ ┌────▼──────┐                  │
│              │ MediaPipe  │ │ OpenPose  │                  │
│              │ (Model1)   │ │ (Model2)  │                  │
│              └────────────┘ └───────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### Option A: Dual Landmarker System (Recommended)

**Concept**: Run both MediaPipe and OpenPose, switch based on active model

**Pros**:
- ✅ Model1 keeps using MediaPipe (fast, works great)
- ✅ Model2 gets OpenPose (accurate, what it was trained on)
- ✅ Each model uses its optimal keypoint extractor

**Cons**:
- More complex code
- Higher resource usage when switching models
- Need to manage two pose estimation systems

**Implementation**:
1. Add OpenPose as optional dependency
2. Create `OpenPoseLandmarker` class
3. Landmarker switches based on `config.input.landmark_source`
4. Model1 config: `"landmark_source": "mediapipe_hands"`
5. Model2 config: `"landmark_source": "openpose"`

### Option B: OpenPose Only for Model2

**Concept**: Only initialize OpenPose when Model2 is active

**Pros**:
- ✅ Simpler - only one landmarker active at a time
- ✅ Lower resource usage
- ✅ Cleaner architecture

**Cons**:
- Switching models requires reinitializing landmarker
- Slight delay when switching

**Implementation**:
1. Detect `landmark_source` from model config
2. Initialize appropriate landmarker
3. Reinitialize when switching models

## OpenPose Integration Options

### Option 1: Python OpenPose Wrapper (Recommended)

**Package**: `openpose-python` or `tf-pose-estimation`

**Installation**:
```bash
pip install tf-pose-estimation
# or
pip install openpose-python
```

**Pros**:
- ✅ Pure Python, easy to install
- ✅ No C++ compilation needed
- ✅ Works on CPU (slower but functional)

**Cons**:
- Slower than official OpenPose
- May have compatibility issues

### Option 2: Official OpenPose (Best Accuracy)

**Source**: https://github.com/CMU-Perceptual-Computing-Lab/openpose

**Installation**: Requires building from source

**Pros**:
- ✅ Best accuracy
- ✅ GPU-accelerated
- ✅ Official implementation

**Cons**:
- ❌ Complex installation (CMake, CUDA, cuDNN)
- ❌ Requires GPU for real-time performance
- ❌ Windows build is challenging

### Option 3: MediaPipe Pose + Manual Mapping (Compromise)

**Concept**: Use MediaPipe Pose but with OpenPose-compatible output format

**Pros**:
- ✅ Easy to implement (already have MediaPipe)
- ✅ Fast, CPU-friendly
- ✅ No new dependencies

**Cons**:
- ❌ Still not exactly OpenPose keypoints
- ❌ May not fix Model2's accuracy issues
- ❌ Coordinate system still different

## Recommended Implementation: tf-pose-estimation

### Why tf-pose-estimation?
1. **Pure Python** - Easy to install with pip
2. **CPU Support** - Works without GPU (slower but functional)
3. **OpenPose Compatible** - Produces OpenPose-format keypoints
4. **Maintained** - Active development
5. **TensorFlow-based** - Integrates well with Python ecosystem

### Installation Steps

```bash
# Install tf-pose-estimation
pip install tf-pose-estimation

# Or if that doesn't work, install from source
git clone https://github.com/ildoonet/tf-pose-estimation
cd tf-pose-estimation
pip install -r requirements.txt
python setup.py install
```

### Code Structure

```python
# src/recognition/openpose_landmarker.py

import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

class OpenPoseLandmarker:
    def __init__(self, model='mobilenet_thin'):
        self.estimator = TfPoseEstimator(
            get_graph_path(model),
            target_size=(432, 368)
        )
    
    def process_frame(self, frame):
        """Extract OpenPose keypoints from frame"""
        humans = self.estimator.inference(frame)
        
        if not humans:
            return None, None
        
        # Extract 55 keypoints in OpenPose format
        keypoints_55 = self._extract_55_keypoints(humans[0])
        return keypoints_55, None
    
    def _extract_55_keypoints(self, human):
        """Convert OpenPose human to 55-point format"""
        # 13 body points + 21 left hand + 21 right hand
        points = np.zeros((55, 2), dtype=np.float32)
        
        # Body keypoints (0-12)
        body_parts = human.body_parts
        # ... map OpenPose body parts to 55-point format
        
        # Hand keypoints (13-54)
        # ... extract hand keypoints
        
        return points
```

### Integration with Existing Code

```python
# src/recognition/landmarker.py

def __init__(self):
    self._mediapipe_hands = None
    self._mediapipe_holistic = None
    self._openpose = None  # NEW
    self._landmark_source = "mediapipe_hands"

def init_from_config(self, config: ModelConfig):
    """Initialize based on model config"""
    source = config.input.landmark_source
    
    if source == "openpose":
        self._init_openpose()
    elif source == "mediapipe_holistic":
        self._init_holistic()
    else:
        self._init_hands()

def _init_openpose(self):
    """Initialize OpenPose"""
    from .openpose_landmarker import OpenPoseLandmarker
    self._openpose = OpenPoseLandmarker()
    print("[Landmarker] Initialized OpenPose")

def process_frame(self, frame):
    """Process frame with appropriate landmarker"""
    if self._landmark_source == "openpose":
        return self._process_openpose(frame)
    elif self._landmark_source == "mediapipe_holistic":
        return self._process_holistic(frame)
    else:
        return self._process_hands(frame)
```

### Model2 Configuration Update

```json
{
  "input": {
    "landmark_source": "openpose",
    "input_shape": [1, 55, 100],
    "use_dimensions": 2,
    "normalize": "none"
  }
}
```

## Performance Considerations

### CPU Performance
- **MediaPipe**: ~30 FPS on CPU
- **tf-pose-estimation (CPU)**: ~5-10 FPS
- **Official OpenPose (CPU)**: ~2-5 FPS

### GPU Performance
- **MediaPipe**: ~60+ FPS
- **tf-pose-estimation (GPU)**: ~20-30 FPS
- **Official OpenPose (GPU)**: ~30-50 FPS

### Recommendation for Video Calls
- Use **MediaPipe for Model1** (fingerspelling) - Fast, responsive
- Use **tf-pose-estimation for Model2** (word recognition) - Slower but accurate
- User can choose based on their needs:
  - Fast fingerspelling → Model1
  - Accurate word recognition → Model2 (accept slower FPS)

## Implementation Steps

### Phase 1: Install and Test OpenPose
1. Install tf-pose-estimation
2. Create test script to verify it works
3. Test keypoint extraction on sample video

### Phase 2: Create OpenPoseLandmarker Class
1. Create `src/recognition/openpose_landmarker.py`
2. Implement 55-keypoint extraction
3. Match OpenPose format exactly

### Phase 3: Integrate with Landmarker
1. Add OpenPose initialization to `landmarker.py`
2. Add switching logic based on `landmark_source`
3. Handle cleanup and resource management

### Phase 4: Update Model2 Config
1. Change `landmark_source` to `"openpose"`
2. Verify normalization is `"none"`
3. Test with Model2

### Phase 5: Testing
1. Test Model1 still works (MediaPipe)
2. Test Model2 with OpenPose
3. Test switching between models
4. Verify predictions are correct

## Timeline Estimate

- **Phase 1**: 1-2 hours (installation and testing)
- **Phase 2**: 2-3 hours (OpenPoseLandmarker implementation)
- **Phase 3**: 2-3 hours (integration)
- **Phase 4**: 30 minutes (configuration)
- **Phase 5**: 1-2 hours (testing)

**Total**: 6-10 hours of development time

## Alternative: Quick Test First

Before full implementation, test if OpenPose fixes Model2:

```python
# test_model2_with_openpose.py
import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import onnxruntime as ort
import numpy as np

# Initialize OpenPose
estimator = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))

# Load Model2
sess = ort.InferenceSession('models/sign/model2/wlasl_pose_tgcn.onnx')

# Capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract OpenPose keypoints
    humans = estimator.inference(frame)
    if humans:
        # Extract 55 keypoints
        keypoints = extract_55_keypoints(humans[0])
        
        # Run inference
        # ... (format and run through model)
        
    cv2.imshow('OpenPose Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

This quick test will verify if OpenPose solves Model2's accuracy issues before committing to full integration.

## Conclusion

**Recommended Approach**:
1. Install tf-pose-estimation
2. Create quick test script
3. Verify Model2 works with OpenPose keypoints
4. If successful, implement full dual-landmarker system
5. Keep Model1 on MediaPipe, Model2 on OpenPose

This gives you the best of both worlds:
- Fast fingerspelling (Model1 + MediaPipe)
- Accurate word recognition (Model2 + OpenPose)

---

**Date**: 2026-05-27
**Status**: Implementation Plan Ready
**Next Action**: Install tf-pose-estimation and create test script
