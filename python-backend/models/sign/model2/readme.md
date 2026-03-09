# WLASL Pose-TGCN (2000 Words)

This folder contains the **Word-Level American Sign Language (WLASL)** model integrated for Phase 6 of the SignSpeak pipeline. 

Unlike the character-level PointNet fingerspelling model, this model recognizes **full words** by tracking upper body, face, and hand movements over a sequence of time.

## 📁 Required Files
For this model to work in SignSpeak, this directory must contain:

1. `wlasl_pose_tgcn.onnx` — The ONNX-converted model weights.
2. `model.json` — The Phase 6 Model Registry configuration.
3. `labels.json` — The mapping of 2000 integer classes to their actual English words.
4. `README.md` — This file.

## 🧠 Model Architecture
* **Original Framework:** PyTorch (converted to ONNX Opsets 13/18)
* **Architecture:** Multi-Attention Spatial-Temporal Graph Convolutional Network (GCN)
* **Dataset:** WLASL2000 (Top 2000 most common ASL words)

## 📡 Input Pipeline (Sequence-Based)
This model requires a rolling buffer of frames. The SignSpeak `PipelineManager` handles this automatically based on the `model.json` configuration.

* **Landmark Source:** `mediapipe_holistic`
* **Nodes Extracted (55 total):**
  * 13 Upper Body / Face points (Nose, shoulders, elbows, wrists, eyes, ears)
  * 21 Left Hand points
  * 21 Right Hand points
* **Sequence Length:** 50 frames of history.
* **Input Shape:** `[1, 55, 100]` *(Batch of 1, 55 spatial nodes, 50 frames × 2 (x,y) coordinates)*

## ⚙️ Post-Processing
Because this predicts full words instead of letters, the `model.json` disables the spell-corrector and misrecognition fixes used by the fingerspelling model. It relies instead on confidence smoothing and a stability gate to trigger the Text-to-Speech (TTS) engine.

## 🔗 Credits
* **Original Paper:** *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison* (WACV 2020)
* **Authors:** Dongxu Li, Cristian Rodriguez, Xin Yu, Hongdong Li
* **Repository:** [dxli94/WLASL](https://github.com/dxli94/WLASL)
