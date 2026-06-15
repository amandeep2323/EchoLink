"""Convert the TF SavedModel to OpenVINO IR using the TF frontend (run in .venv).
Tests whether the GRU compiles natively (GRUSequence) instead of an ONNX Loop.
"""
import os
import numpy as np
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
SM_DIR = os.path.join(HERE, "saved_model")
IR_PATH = os.path.join(HERE, "model4.xml")

print("[ir] Converting SavedModel -> OpenVINO IR (TF frontend)...")
ov_model = ov.convert_model(SM_DIR, input=[("keypoints", [1, 150, 172])])
ov.save_model(ov_model, IR_PATH)
print("[ir] Saved", IR_PATH)

print("[ir] Compiling + test inference...")
core = ov.Core()
cm = core.compile_model(ov_model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
ir = cm.create_infer_request()
x = np.random.randn(1, 150, 172).astype("float32")
r = ir.infer({cm.inputs[0]: x})
o = list(r.values())[0]
print("[ir] OpenVINO IR OK output", o.shape, "argmax", int(o.argmax()))
