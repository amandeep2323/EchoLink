# Model3: LSTM ASL Recognition (250 Signs)

## Important Update (2026-05-31)

Model3 input semantics were corrected after validating the original source code and ONNX model:

- Correct runtime input: `[T, 543, 3]`
- `T` is temporal sequence length (default: 30 frames in this app)
- `543` is the full holistic landmark set per frame (468 face + 33 pose + 21 left hand + 21 right hand)
- `3` is `(x, y, z)` coordinates

This replaces the earlier incorrect interpretation that the model expected 543 temporal frames with only 3 aggregate features.

## Overview

Model3 is an LSTM-based American Sign Language (ASL) recognition model that recognizes 250 common ASL signs with 87% accuracy on the training dataset. The model uses temporal sequence classification to analyze hand, pose, and face landmarks over time.

## Model Information

- **Architecture**: LSTM (Long Short-Term Memory) neural network
- **Source**: [jamesjbustos/sign-language-recognition](https://github.com/jamesjbustos/sign-language-recognition)
- **Number of Classes**: 250 ASL signs
- **Accuracy**: 87% on training dataset
- **Model Type**: Word-level sign language recognition
- **Format**: ONNX (converted from TensorFlow Lite)

## Input Requirements

### Landmark Source
- **Framework**: MediaPipe Holistic
- **Components**:
  - 468 face landmarks
  - 33 pose landmarks (upper body)
  - 21 left hand landmarks
  - 21 right hand landmarks
- **Total**: 543 landmarks × 3 coordinates (x, y, z) per frame

### Input Shape
- **Tensor Shape**: `[30, 543, 3]`
  - Sequence length: 30 frames (~1 second at 30fps)
  - Landmarks: 543 points per frame
  - Features: 3 (x, y, z coordinates per landmark)
- **Data Type**: float32
- **Normalization**: Min-max normalization to [0, 1] range

### Preprocessing
1. Extract MediaPipe Holistic landmarks from each frame
2. Flatten landmarks into feature vectors (x, y, z coordinates)
3. Apply min-max normalization to each frame
4. Buffer 30 consecutive frames (~1 second)
5. Stack into sequence tensor [30, 543, 3]

## Output Format

### Output Shape
- **Tensor Shape**: `[1, 250]`
  - Batch size: 1
  - Classes: 250 sign predictions
- **Data Type**: float32
- **Format**: Logits (softmax applied by model loader)

### Label Mapping
- Labels are mapped in `labels.json`
- Indices 0-249 correspond to 250 ASL signs
- Common signs include: hello, thank you, please, yes, no, etc.
- Full vocabulary includes everyday words, animals, colors, family members, and more

## Inference Type

### Sequence-Based (Temporal)
- **Type**: Temporal sequence classification
- **Sequence Length**: 30 frames (~1 second at 30fps)
- **Stride**: 1 frame (sliding window)
- **Buffering**: Requires maintaining a rolling buffer of 30 frames
- **Latency**: ~33ms per frame + inference time

### Inference Process
1. Capture video frame
2. Extract MediaPipe landmarks
3. Add frame to sequence buffer
4. When buffer reaches 30 frames, run inference
5. Slide window by 1 frame and repeat

## Model Conversion

### Original Format
- **Source Format**: TensorFlow Lite (.tflite)
- **Original File**: `model.tflite` (3.2 MB)
- **Downloaded From**: GitHub repository weights/ directory

### Conversion to ONNX
- **Converter**: tf2onnx
- **ONNX Opset**: 13
- **Conversion Script**: `convert_model3.py` (in python-backend/)
- **Verification**: Input/output shape validation and numerical accuracy check

### Conversion Command
```bash
python convert_model3.py models/sign/model3/staging/model.tflite --output-dir models/sign/model3/
```

**Note**: Requires Python 3.11 or 3.12 (TensorFlow not available for Python 3.14+). See `CONVERSION_INSTRUCTIONS.md` for alternative conversion methods.

## Performance Characteristics

### Inference Performance
- **Inference Latency**: ~50-100ms per sequence (CPU)
- **Memory Usage**: ~50-100 MB (model + sequence buffer)
- **Recommended Hardware**: CPU (GPU optional for faster inference)
- **Frame Rate**: 30 fps (real-time capable)

### Accuracy Characteristics
- **Training Accuracy**: 87%
- **Best Performance**: Full body visibility with good lighting
- **Degradation Factors**:
  - Partial occlusion of hands or body
  - Poor lighting conditions
  - Fast or exaggerated movements
  - Camera angle variations

## Known Limitations

### Model Limitations
1. **Vocabulary Size**: Limited to 250 signs (subset of ASL)
2. **Temporal Dependency**: Requires 30 consecutive frames (1 second)
3. **Gesture Overlap**: Some signs have similar gestures and may be confused
4. **Static Signs**: May not distinguish well between similar static handshapes

### Technical Limitations
1. **Sequence Buffer**: Adds ~1-second latency before first prediction
2. **Full Body Required**: Needs visibility of upper body, hands, and face
3. **Lighting Sensitive**: MediaPipe landmark detection degrades in poor lighting
4. **Single Signer**: Trained on single-signer data, may not generalize well
5. **Temporal Buffer**: Sequence length is currently 30 frames; shorter windows may reduce accuracy

### Environmental Requirements
1. **Camera**: Minimum 30fps, 640x480 resolution recommended
2. **Lighting**: Good ambient lighting for landmark detection
3. **Background**: Uncluttered background improves landmark accuracy
4. **Distance**: Signer should be 1-2 meters from camera

## Integration with Application

### Model Registry
- Model3 is automatically discovered by `ModelRegistry` on startup
- Directory structure follows model1/model2 patterns
- Configuration loaded from `model.json`

### Model Selection
- Selectable via UI or by updating `_active_model.txt` to "model3"
- Model persists across application restarts
- Switching models unloads previous model and loads model3

### Recognition Pipeline
- Integrates seamlessly with existing recognition pipeline
- Uses same `ModelLoader` interface as model1 and model2
- No code changes required in core components

## Usage Example

### Selecting Model3
1. Start the application
2. Open model selection UI
3. Select "LSTM ASL Recognition (250 Signs)"
4. Model3 is now active for recognition

### Recognition Workflow
1. Position yourself in front of camera (full upper body visible)
2. Perform ASL sign clearly
3. Hold the sign briefly (~1 second / 30 frames)
4. Model predicts sign and displays result
5. Confidence score shown with prediction

### Tips for Best Results
- Ensure good lighting
- Keep hands and upper body in frame
- Perform signs at normal speed (not too fast)
- Hold signs for the full sequence length (~1 second)
- Avoid cluttered backgrounds
- Maintain consistent distance from camera

## Files in This Directory

- **model.json**: Model configuration file (metadata, input/output specs, inference settings)
- **model.onnx**: ONNX model file (✅ Converted and ready)
- **labels.json**: Label map (250 sign names indexed 0-249)
- **README.md**: This documentation file

## Troubleshooting

### Model Not Appearing in UI
- Check that `model.json` exists and is valid JSON
- Verify `labels.json` exists with 250 labels
- Ensure `model.onnx` exists (✅ Already converted)
- Restart application to trigger model discovery

### Low Accuracy / No Predictions
- Ensure good lighting conditions
- Check that full upper body is visible
- Verify MediaPipe landmarks are being detected
- Hold signs for the full sequence length (~1 second)
- Try adjusting sequence_length in `model.json` for shorter sequences
- Increase confidence threshold in `model.json` if getting false positives

### Conversion Errors
- ✅ Conversion already complete
- Model successfully converted using Google Colab
- ONNX file verified and working

### Performance Issues
- Reduce sequence_length in `model.json` (trade-off: accuracy)
- Increase stride to skip frames (trade-off: temporal resolution)
- Use GPU acceleration if available (install onnxruntime-gpu)
- Close other applications to free up resources

## References

- **Original Repository**: https://github.com/jamesjbustos/sign-language-recognition
- **MediaPipe Holistic**: https://google.github.io/mediapipe/solutions/holistic
- **ONNX Runtime**: https://onnxruntime.ai/
- **tf2onnx**: https://github.com/onnx/tensorflow-onnx

## Version History

- **v1.0** (2026-05-27): Initial integration
  - Downloaded model from GitHub repository
  - Converted to ONNX format using Google Colab
  - Created configuration files
  - Integrated with model registry
  - ✅ Model fully functional and ready for use

## License

Model3 is based on the jamesjbustos/sign-language-recognition repository. Please refer to the original repository for licensing information.

## Contact

For issues or questions about model3 integration, please refer to:
- Original model: https://github.com/jamesjbustos/sign-language-recognition
- Application issues: Contact application maintainers
