"""Integration verification for the Model 4 integration (run from python-backend).

    ..\.venv\Scripts\python.exe verify_model4.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.model_registry import ModelRegistry
from src.models.model_config import ModelConfig
from src.models.model_loader import ModelLoader

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models", "sign")

ok = True

# 1) Registry discovers all models incl. model4
reg = ModelRegistry(MODELS_DIR)
reg.discover()
ids = reg.model_ids
print("[verify] discovered:", ids)
assert "model4" in ids, "model4 not discovered"
for m in ("model1", "model2", "model3"):
    assert m in ids, f"regression: {m} missing"

# 2) model4 config + label count
cfg = reg.get_model_by_id("model4")
print(f"[verify] model4: name='{cfg.name}', type={cfg.type}, classes={cfg.num_classes}, "
      f"backend={cfg.inference.backend}, feature_mode={cfg.input.feature_mode}, "
      f"tensor_format={cfg.inference.tensor_format}")
assert cfg.num_classes == 2731, f"expected 2731 labels, got {cfg.num_classes}"
assert cfg.input.feature_mode == "asl_citizen_86"

# 3) Loader routes model4 to OpenVINO and runs inference
loader = ModelLoader()
loader.load_from_config(cfg)
print(f"[verify] loaded backend={loader.backend}, input_shape={loader.input_shape}")
assert loader.backend == "openvino", f"expected openvino backend, got {loader.backend}"
x = np.random.randn(1, 150, 172).astype(np.float32)
probs = loader.predict_raw(x)
probs = np.asarray(probs).reshape(-1)
print(f"[verify] inference output dim={probs.shape}, argmax={int(probs.argmax())}, "
      f"sum={float(probs.sum()):.3f}")
assert probs.shape[0] == 2731

# 4) Recognizer tensor build invariant
from src.recognition.recognizer import Recognizer
rec = Recognizer.__new__(Recognizer)
rec._target_frames = 150
for T in (1, 48, 150):
    stacked = np.random.randn(T, 86, 2).astype(np.float32)
    t = rec._build_asl_citizen_tensor(stacked)
    assert t.shape == (1, 150, 172), f"T={T} -> {t.shape}"
print("[verify] tensor build (1,150,172) OK for T in {1,48,150}")

# 5) Models 1-3 still load
for mid in ("model1", "model2", "model3"):
    c = reg.get_model_by_id(mid)
    l = ModelLoader()
    l.load_from_config(c)
    print(f"[verify] {mid} loaded backend={l.backend}")
    if mid == "model2":
        assert l.backend in ("openvino", "signbart"), f"model2 backend regressed: {l.backend}"

print("\n[verify] ALL CHECKS PASSED")
