"""
Local Model 4 (ASL-Citizen GRU, 2731 classes) test harness.
============================================================
Adapted from the user's Colab `model_test.txt` to run on this machine.

Differences from the Colab script:
  - Runs inference through ONNX Runtime (model.onnx) instead of Keras,
    so it works in the app's `.venv` (which has onnxruntime + mediapipe
    but NOT tensorflow). Falls back to Keras if onnxruntime model missing
    and tensorflow is available.
  - Reads test clips from the local `test_vidoes/` folder. The video file
    name (minus extension) is the ground-truth gloss.
  - Prints per-video predicted gloss + confidence and a final accuracy.

Preprocessing mirrors the author's pipeline exactly:
  86 landmarks (pose12 + Lhand21 + Rhand21 + face32), x/y only,
  per-frame anchor normalization, tile/pad to 150 frames,
  reshape to (150, 172), batch -> (1, 150, 172).

Usage (from python-backend/models/sign/model4/):
    ..\..\..\..\.venv\Scripts\python.exe test_model4_local.py
"""

import os
import re
import glob
import json
import numpy as np
import cv2
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
IR_PATH = os.path.join(HERE, "model4.xml")
ONNX_PATH = os.path.join(HERE, "model.onnx")
KERAS_PATH = os.path.join(HERE, "best_model_2731.keras")
DECODER_JSON = os.path.join(HERE, "index_to_gloss_2731.json")
TEST_DIR = os.path.join(HERE, "test_vidoes")

PADDED_LEN = 150


# ── Preprocessing (author's exact logic) ───────────────────────────────

def hand_normalize(hand_data):
    xmin, xmax = np.min(hand_data[:, 0]), np.max(hand_data[:, 0])
    ymin, ymax = np.min(hand_data[:, 1]), np.max(hand_data[:, 1])
    width, height = xmax - xmin, ymax - ymin
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    if width != 0:
        hand_data[:, 0] = (hand_data[:, 0] - center[0]) / width
    else:
        hand_data[:, 0] = 0
    if height != 0:
        hand_data[:, 1] = (hand_data[:, 1] - center[1]) / height
    else:
        hand_data[:, 1] = 0
    return hand_data


def distance(x1, x2):
    delta = x1 - x2
    return (delta[0] ** 2 + delta[1] ** 2) ** 0.5


def anchor_norm(target, scale, reference):
    return (target - reference) / (scale + 0.01)


def pad_video(X):
    data_array = np.zeros((PADDED_LEN, X.shape[1], X.shape[2]))
    for landmark in range(X.shape[1]):
        for coord in range(X.shape[2]):
            data_array[:, landmark, coord] = np.tile(
                X[:, landmark, coord], int(PADDED_LEN / X.shape[0] + 2)
            )[:PADDED_LEN]
    return data_array


IMPORTANT_FACE = [
    33, 133, 159, 145, 153, 144, 362, 263, 386, 374, 380, 373,
    70, 63, 105, 66, 107, 295, 282, 320, 285, 318,
    1, 168, 197, 4, 78, 308, 13, 14, 81, 311,
]
POSE_REMOVE = set(list(range(23, 33)) + list(range(0, 11)))
POSE_KEEP = [i for i in range(33) if i not in POSE_REMOVE]


def extract_landmarks(results):
    pose = (
        [(lm.x, lm.y) for i, lm in enumerate(results.pose_landmarks.landmark) if i in POSE_KEEP]
        if results.pose_landmarks else [(0, 0)] * len(POSE_KEEP)
    )
    left_hand = (
        [(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark]
        if results.left_hand_landmarks else [(0, 0)] * 21
    )
    right_hand = (
        [(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark]
        if results.right_hand_landmarks else [(0, 0)] * 21
    )
    face = (
        [(results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y) for i in IMPORTANT_FACE]
        if results.face_landmarks else [(0, 0)] * len(IMPORTANT_FACE)
    )
    # Order: pose(12) + Lhand(21) + Rhand(21) + face(32) = 86
    return np.array(pose + left_hand + right_hand + face)


def preprocess_frame(X):
    neck = abs(X[0] + X[1]) / 2
    X[:] = anchor_norm(X[:], distance(X[0], X[1]), neck)
    X[54:] = anchor_norm(X[54:], distance(X[0], X[1]), X[79])
    left_arm, right_arm = [2, 4, 6, 8, 10], [3, 5, 7, 9, 11]
    X[left_arm] = anchor_norm(X[left_arm], distance(X[0], X[2]), X[0])
    X[right_arm] = anchor_norm(X[right_arm], distance(X[1], X[3]), X[1])
    X[12:33] = hand_normalize(X[12:33])
    X[33:54] = hand_normalize(X[33:54])
    return X


def normalize_for_compare(s):
    return re.sub(r"[^A-Za-z]", "", s).upper()


# ── Inference backends ──────────────────────────────────────────────────

class OpenVinoRunner:
    def __init__(self, path):
        import openvino as ov
        core = ov.Core()
        self.cm = core.compile_model(path, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        self.ir = self.cm.create_infer_request()
        self.iport = self.cm.inputs[0]
        print(f"[test] OpenVINO IR backend — input {self.iport.get_any_name()} {self.iport.shape}")

    def predict(self, x):
        r = self.ir.infer({self.iport: x.astype(np.float32)})
        return list(r.values())[0][0]


class OnnxRunner:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        print(f"[test] ONNX backend — input '{self.iname}' shape {self.sess.get_inputs()[0].shape}")

    def predict(self, x):
        return self.sess.run(None, {self.iname: x.astype(np.float32)})[0][0]


class KerasRunner:
    def __init__(self, path):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        print("[test] Keras backend")

    def predict(self, x):
        return self.model.predict(x, verbose=0)[0]


def make_runner():
    if os.path.exists(IR_PATH):
        try:
            return OpenVinoRunner(IR_PATH)
        except Exception as e:
            print(f"[test] OpenVINO IR load failed ({e}); trying ONNX...")
    if os.path.exists(ONNX_PATH):
        try:
            return OnnxRunner(ONNX_PATH)
        except Exception as e:
            print(f"[test] ONNX load failed ({e}); trying Keras...")
    return KerasRunner(KERAS_PATH)


def main():
    with open(DECODER_JSON, "r") as f:
        index_to_gloss = {int(k): v for k, v in json.load(f).items()}

    runner = make_runner()

    videos = sorted(glob.glob(os.path.join(TEST_DIR, "*.mp4")))
    if not videos:
        print(f"[test] No videos in {TEST_DIR}")
        return

    mp_holistic = mp.solutions.holistic
    correct = 0
    total = 0

    for video_path in videos:
        name = os.path.splitext(os.path.basename(video_path))[0]
        expected = normalize_for_compare(name)
        cap = cv2.VideoCapture(video_path)
        frames = []
        with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                lm = extract_landmarks(results)
                frames.append(preprocess_frame(lm))
        cap.release()

        if not frames:
            print(f"{name:24s} -> UNKNOWN (no frames)")
            total += 1
            continue

        arr = np.array(frames, dtype=np.float32)           # (F, 86, 2)
        padded = pad_video(arr)                            # (150, 86, 2)
        flat = padded.reshape(padded.shape[0], -1)         # (150, 172)
        model_input = np.expand_dims(flat, axis=0)         # (1, 150, 172)

        probs = runner.predict(model_input)
        top5 = np.argsort(probs)[::-1][:5]
        pred = index_to_gloss.get(int(top5[0]), "UNKNOWN")
        conf = float(probs[top5[0]])
        hit = normalize_for_compare(pred) == expected
        correct += int(hit)
        total += 1
        top5_str = ", ".join(f"{index_to_gloss.get(int(i), '?')}({probs[i]:.2f})" for i in top5)
        print(f"{name:24s} -> {pred:20s} {conf:.3f} {'OK' if hit else '  '}  | top5: {top5_str}")

    print(f"\n[test] Accuracy: {correct}/{total} = {correct / total:.1%}")


if __name__ == "__main__":
    main()
