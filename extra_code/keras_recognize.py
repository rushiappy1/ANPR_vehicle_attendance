#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Optional: force CPU and reduce TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU, avoid cuInit 303 spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN info spam


# ----------------------------
# SAME CHARACTER SET AS TRAINING
# ----------------------------
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
PAD_VALUE = 0
BLANK_INDEX = len(CHARS)      # 36
NUM_CHARS = len(CHARS) + 1    # 37


# ----------------------------
# CUSTOM OBJECTS FROM TRAINING
# ----------------------------

# Dummy masked_sparse_ce just to satisfy loader.
# For inference, loss is never used.
def masked_sparse_ce(y_true, y_pred):
    return tf.constant(0.0, dtype=tf.float32)


# If you have the REAL implementation from training, put it here instead,
# but keep the same function name.


# ----------------------------
# DEBUG: visualization (optional)
# ----------------------------
def show_image(img):
    plt.imshow(img, cmap="gray")
    plt.title("Preprocessed Image")
    plt.show()


# ----------------------------
# CTC Decoding
# ----------------------------
def ctc_decode(pred):
    decoded, _ = tf.keras.backend.ctc_decode(
        pred,
        input_length=np.ones(pred.shape[0]) * pred.shape[1],
        greedy=True,
    )
    sequence = decoded[0].numpy()[0]

    print("Decoded raw sequence:", sequence)  # DEBUG

    text = "".join(
        IDX2CHAR[i] for i in sequence
        if 0 < i < BLANK_INDEX
    )
    return text


# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(img_path, img_w, img_h, debug=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")

    img = cv2.resize(img, (img_w, img_h))
    img = img.astype("float32") / 255.0

    if debug:
        show_image(img)

    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    img = np.expand_dims(img, axis=0)   # (1, H, W, 1)
    return img


# ----------------------------
# MAIN RECOGNITION FUNCTION
# ----------------------------
def recognize(model_path, image_path, debug=False):
    print("Loading model...")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "masked_sparse_ce": masked_sparse_ce,
            # if you also had a custom ctc loss name, add here as well
        },
        compile=False,  # do NOT compile; we just want inference
    )

    print("Model loaded.")

    # Auto-detect correct image size
    input_h = model.input_shape[1]
    input_w = model.input_shape[2]
    print(f"Model expects size: {input_w}×{input_h}")

    # Preprocess
    img = preprocess_image(image_path, input_w, input_h, debug=debug)

    print("Running prediction...")
    pred = model.predict(img)

    print("Prediction logits shape:", pred.shape)

    # Decode CTC
    text = ctc_decode(pred)

    print("\nRecognized Text:", text)
    return text


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python keras_recognize.py <model.keras> <image>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.isfile(model_path):
        print(f"[ERR] Model not found: {model_path}")
        sys.exit(1)

    if not os.path.isfile(image_path):
        print(f"[ERR] Image not found: {image_path}")
        sys.exit(1)

    recognize(model_path, image_path, debug=False)
