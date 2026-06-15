"""Export the Keras GRU model to a TF SavedModel (run in model4_tf_venv).
Then OpenVINO's TF frontend can convert it to IR with native GRUSequence ops,
avoiding the ONNX 'Loop' operator that OpenVINO cannot compile.
"""
import os
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH = os.path.join(HERE, "best_model_2731.keras")
SM_DIR = os.path.join(HERE, "saved_model")

model = tf.keras.models.load_model(KERAS_PATH)

@tf.function(input_signature=[tf.TensorSpec((1, 150, 172), tf.float32, name="keypoints")])
def serve(x):
    return {"output": model(x)}

tf.saved_model.save(model, SM_DIR, signatures={"serving_default": serve})
print("[savedmodel] Saved to", SM_DIR)
