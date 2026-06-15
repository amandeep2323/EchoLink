"""
Convert Model 4 (ASL-Citizen GRU) Keras -> ONNX.
=================================================
Run with the isolated TF environment that has tensorflow-cpu + tf2onnx:

    ..\..\..\..\model4_tf_venv\Scripts\python.exe convert_keras_to_onnx.py

Produces model.onnx (input: float32 [1, 150, 172], output: softmax [1, 2731]).
This is a test/staging converter; the spec will formalize it as convert_model4.py.
"""

import os
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH = os.path.join(HERE, "best_model_2731.keras")
ONNX_PATH = os.path.join(HERE, "model.onnx")

SEQ_LEN = 150
FEATURES = 172


def main():
    print(f"[convert] Loading {KERAS_PATH}")
    model = tf.keras.models.load_model(KERAS_PATH)
    model.summary()

    in_shape = model.input_shape
    out_shape = model.output_shape
    print(f"[convert] Keras input_shape={in_shape}, output_shape={out_shape}")

    spec = (tf.TensorSpec((1, SEQ_LEN, FEATURES), tf.float32, name="keypoints"),)

    @tf.function(input_signature=spec)
    def serve(x):
        return model(x)

    print("[convert] Running tf2onnx...")
    onnx_model, _ = tf2onnx.convert.from_function(
        serve, input_signature=spec, opset=13, output_path=ONNX_PATH
    )
    print(f"[convert] Saved {ONNX_PATH}")

    # Parity check
    print("[convert] Verifying numerical parity...")
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, SEQ_LEN, FEATURES)).astype(np.float32)

    keras_out = model.predict(sample, verbose=0)
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {iname: sample})[0]

    max_abs = float(np.max(np.abs(keras_out - onnx_out)))
    same_argmax = int(np.argmax(keras_out)) == int(np.argmax(onnx_out))
    print(f"[convert] max abs diff = {max_abs:.3e}, argmax match = {same_argmax}")
    print("[convert] DONE")


if __name__ == "__main__":
    main()
