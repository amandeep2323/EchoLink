# Model 2 — SignBart WLASL (1000 Words)

Word-level American Sign Language recognition using **SignBart**, a lightweight
(~4.4M param) BART-style encoder–decoder that operates on MediaPipe Holistic
skeleton sequences. It decouples the x and y coordinates (x → encoder,
y → decoder) and fuses them with cross-attention.

This replaces the previous OpenPose-based WLASL Pose-TGCN model, which was
CPU-heavy (~91% CPU, 1–2 FPS). SignBart runs on MediaPipe Holistic (no OpenPose
dependency) and is far lighter.

- **Paper / source**: https://github.com/TinhNguyen2312/SignBart (arXiv 2506.21592)
- **Pretrained weights**: WLASL-1000 (`WLASL-1000.pth`) from
  https://www.kaggle.com/models/nguyenchitinh/signbart
- **Classes**: 1000 WLASL words (`labels.json`)

## Files

| File | Purpose |
|------|---------|
| `signbart_wlasl1000.onnx` | Exported SignBart model (dual input: keypoints + attention_mask) |
| `labels.json` | Ordered list of 1000 WLASL gloss words |
| `model.json` | EchoLink model configuration |

## How it works in EchoLink

1. **Landmarks** — `landmark_source: mediapipe_holistic`, `feature_mode: signbart_holistic75`.
   The landmarker emits a `(75, 2)` array per frame: 33 pose + 21 left-hand +
   21 right-hand keypoints (x, y), zero-filled where a part is missing.
2. **Buffering** — the recognizer buffers `sequence_length` (48) frames.
3. **Normalization** — per-part bounding-box normalization (body / left hand /
   right hand) is applied in `recognizer._build_signbart_tensor`, exactly
   matching SignBart's training `dataset.py`.
4. **Inference** — `backend: signbart` in `model_loader.py` runs the dual-input
   ONNX via ONNX Runtime: `keypoints (1,T,75,2)` + `attention_mask (1,T)` →
   `logits (1,1000)` → softmax → top-k → word.

## Regenerating the ONNX

From `SignBart-main/` (with its venv):

```
venv\Scripts\python.exe export_signbart_onnx.py ^
    --config configs/WLASL-1000.yaml ^
    --weights pretrained_models/WLASL-1000.pth ^
    --out signbart_wlasl1000.onnx
```

Then copy `signbart_wlasl1000.onnx` and `labels_wlasl/labels_list.json`
(as `labels.json`) into this folder.

## Label ordering note

`labels.json` is the first 1000 glosses of `WLASL_v0.3.json` in file order — the
standard WLASL subset convention. If live predictions map to the wrong words,
the label ordering used during training differs and this file must be
regenerated to match.
