# ============================================================
# SignSpeak — Kaggle ONNX Conversion Notebook
# ============================================================
# 
# HOW TO USE ON KAGGLE:
# 
# 1. Go to https://www.kaggle.com/code
# 2. Click "New Notebook"
# 3. Upload your model file:
#    - Click the "+" button on the right sidebar → "Upload" → select your model5.keras (or .h5)
#    - It will appear at: /kaggle/input/your-dataset-name/model5.keras
#    - OR just upload directly to the notebook's working directory
# 4. Copy-paste ALL of this code into a single cell
# 5. Click "Run All"
# 6. Download the .onnx file from the output
#
# ALTERNATIVE (easier upload):
# 1. In the notebook, use the file upload cell below
# ============================================================

# %% Cell 1 — Install dependencies (Kaggle already has TF + numpy, just need tf2onnx)
import subprocess
import sys

print("=" * 60)
print("  Step 1: Installing tf2onnx")
print("=" * 60)

subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx", "onnx", "onnxruntime", "-q"])
print("✓ tf2onnx installed")

# %% Cell 2 — Upload model (if not using Kaggle datasets sidebar)
import os

# === OPTION A: If you uploaded via Kaggle sidebar ===
# Change this path to match your upload:
# MODEL_PATH = "/kaggle/input/your-dataset-name/model5.keras"

# === OPTION B: If you uploaded to working directory ===  
# MODEL_PATH = "/kaggle/working/model5.keras"

# === OPTION C: Auto-find the model ===
MODEL_PATH = None

def find_model():
    """Auto-search for .h5 or .keras files"""
    search_dirs = [
        "/kaggle/working",
        "/kaggle/input",
        ".",
    ]
    extensions = [".keras", ".h5"]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                for ext in extensions:
                    if f.endswith(ext):
                        return os.path.join(root, f)
    return None

if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
    MODEL_PATH = find_model()

if MODEL_PATH is None:
    print("=" * 60)
    print("  ❌ No model file found!")
    print("=" * 60)
    print()
    print("  Upload your model using one of these methods:")
    print()
    print("  Method 1 (Kaggle sidebar):")
    print("    Click '+' → 'Upload' → select model5.keras")
    print()
    print("  Method 2 (Code upload):")
    print("    Add this code BEFORE this cell:")
    print("    ─────────────────────────────────")
    print("    from google.colab import files  # or use Kaggle file upload")
    print("    uploaded = files.upload()        # opens file picker")
    print("    ─────────────────────────────────")
    print()
    print("  Method 3 (Direct URL):")
    print("    !wget YOUR_MODEL_URL -O model5.keras")
    print()
    raise FileNotFoundError("Upload your .h5 or .keras model first")

print(f"✓ Found model: {MODEL_PATH}")
print(f"  Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")

# %% Cell 3 — Load model and print info
print()
print("=" * 60)
print("  Step 2: Loading model")
print("=" * 60)

import numpy as np

# Handle keras/tensorflow imports
try:
    import keras
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Loaded with keras {keras.__version__}")
except Exception as e1:
    try:
        from tensorflow import keras as tf_keras
        model = tf_keras.models.load_model(MODEL_PATH)
        print(f"✓ Loaded with tensorflow.keras")
        keras = tf_keras
    except Exception as e2:
        print(f"❌ Failed to load model:")
        print(f"   keras error: {e1}")
        print(f"   tf.keras error: {e2}")
        raise

print()
print("  Model Info:")
print(f"  ├── Name:         {model.name}")
print(f"  ├── Input shape:  {model.input_shape}")
print(f"  ├── Output shape: {model.output_shape}")
print(f"  ├── Parameters:   {model.count_params():,}")
print()
model.summary()

# %% Cell 4 — Verify model works
print()
print("=" * 60)
print("  Step 3: Verifying model inference")
print("=" * 60)

# Create dummy input matching model's expected shape
input_shape = model.input_shape
# Replace None (batch dim) with 1
test_shape = tuple(1 if d is None else d for d in input_shape)
dummy_input = np.random.rand(*test_shape).astype(np.float32)

print(f"  Test input shape: {dummy_input.shape}")
predictions = model.predict(dummy_input, verbose=0)
print(f"  Test output shape: {predictions.shape}")
print(f"  Test output (first 5): {predictions[0][:5]}")

# For SignSpeak PointNet: check if model needs 2 or 3 dims
if input_shape == (None, 21, 3):
    print()
    print("  PointNet model detected — testing input dimensions:")
    print(f"  ✓ Full (x,y,z) inference works: shape {dummy_input.shape}")
    
    # Test if model also accepts x,y only
    try:
        dummy_xy = dummy_input[:, :, :2]
        pred_xy = model.predict(dummy_xy, verbose=0)
        print(f"  ✓ Reduced (x,y) inference also works: shape {dummy_xy.shape}")
        print(f"  → Model accepts both 2 and 3 dims")
    except Exception as e:
        print(f"  ✗ Reduced (x,y) inference fails: {type(e).__name__}")
        print(f"  → Model REQUIRES all 3 dims (x,y,z) — Conv1D architecture")
        print(f"  → SignSpeak will auto-detect this and use 3 dims")

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
pred_idx = np.argmax(predictions[0])
print(f"  Predicted class: {pred_idx} → '{LETTERS[pred_idx] if pred_idx < len(LETTERS) else '?'}'")
print(f"  Confidence: {predictions[0][pred_idx]:.4f}")
print("✓ Model inference works")

# %% Cell 5 — Convert to ONNX
print()
print("=" * 60)
print("  Step 4: Converting to ONNX")
print("=" * 60)

import tf2onnx
import tensorflow as tf

# Output path
model_basename = os.path.splitext(os.path.basename(MODEL_PATH))[0]
ONNX_PATH = f"/kaggle/working/{model_basename}.onnx"

print(f"  Source: {MODEL_PATH}")
print(f"  Target: {ONNX_PATH}")
print(f"  Converting... (this may take 1-2 minutes)")
print()

# Get the input spec from the model
input_shape = model.input_shape
spec = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")]

# Convert
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=ONNX_PATH,
)

onnx_size = os.path.getsize(ONNX_PATH) / 1024 / 1024
print()
print(f"✓ ONNX model saved: {ONNX_PATH}")
print(f"  Size: {onnx_size:.2f} MB")

# %% Cell 6 — Verify ONNX model
print()
print("=" * 60)
print("  Step 5: Verifying ONNX model")
print("=" * 60)

import onnx
import onnxruntime as ort

# Load and check ONNX model
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("✓ ONNX model validation passed")

# Run inference with ONNX Runtime
session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
input_shape_onnx = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name

print(f"  ONNX input:  {input_name} → {input_shape_onnx}")
print(f"  ONNX output: {output_name}")

# Test with same dummy input
onnx_pred = session.run([output_name], {input_name: dummy_input})[0]
print(f"  ONNX output shape: {onnx_pred.shape}")

# Compare Keras vs ONNX predictions
keras_pred = model.predict(dummy_input, verbose=0)
max_diff = np.max(np.abs(keras_pred - onnx_pred))
print(f"  Max difference (Keras vs ONNX): {max_diff:.8f}")

if max_diff < 0.001:
    print("✓ ONNX predictions match Keras — conversion successful!")
else:
    print(f"⚠ Predictions differ by {max_diff:.6f} — check model carefully")

pred_idx_onnx = np.argmax(onnx_pred[0])
print(f"  ONNX predicted: {pred_idx_onnx} → '{LETTERS[pred_idx_onnx] if pred_idx_onnx < len(LETTERS) else '?'}'")

# %% Cell 7 — Create label map JSON
print()
print("=" * 60)
print("  Step 6: Creating label map")
print("=" * 60)

import json

labels = {
    "letters": LETTERS,
    "num_classes": len(LETTERS),
    "class_map": {str(i): letter for i, letter in enumerate(LETTERS)},
    "note": "J and Z excluded (require motion gestures)"
}

LABELS_PATH = f"/kaggle/working/labels.json"
with open(LABELS_PATH, "w") as f:
    json.dump(labels, f, indent=2)

print(f"✓ Label map saved: {LABELS_PATH}")
print(f"  Classes: {LETTERS} ({len(LETTERS)} total)")

# %% Cell 8 — Summary and download instructions
print()
print("=" * 60)
print("  ✅ CONVERSION COMPLETE")
print("=" * 60)
print()
print("  Files created:")
print(f"    1. {ONNX_PATH} ({onnx_size:.2f} MB)")
print(f"    2. {LABELS_PATH}")
print()
print("  Download these files and place them in your project:")
print("  ┌─────────────────────────────────────────────────────┐")
print(f"  │  {model_basename}.onnx  →  python-backend/models/sign/  │")
print(f"  │  labels.json      →  python-backend/models/sign/  │")
print("  └─────────────────────────────────────────────────────┘")
print()
print("  On Kaggle: Click the ▶ Output tab on the right")
print("             to see and download the files.")
print()
print("  Your model directory should look like:")
print("    python-backend/models/sign/")
print(f"    ├── {model_basename}.onnx")
print("    └── labels.json")
print()
print("  Then start SignSpeak:")
print("    cd python-backend")
print("    python main.py")
print()
print("=" * 60)
