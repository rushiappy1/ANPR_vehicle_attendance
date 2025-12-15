import os

# -------------------------------------------------
# Force CPU and reduce TensorFlow log spam
# -------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU, avoid cuInit 303 error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # optional: disable oneDNN optimizations info

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# -------------------------------------------------
# Model and input configuration
# -------------------------------------------------
# Your saved CRNN model
MODEL_PATH = os.path.join(os.getcwd(), "plate_crnn_seq.keras")

# From the error: expected shape=(None, 64, 256, 1)
IMG_HEIGHT = 64
IMG_WIDTH = 256
CHANNELS = 1  # grayscale

# Charset used during training – CHANGE if your training alphabet differs
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = len(ALPHABET)  # typical CTC: blank is last index


def load_crnn_model():
    """
    Load the CRNN model from disk for inference.
    """
    model = load_model(MODEL_PATH, compile=False)
    return model


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load one plate image and convert it to model input:
      - Read using OpenCV
      - Resize to (IMG_WIDTH, IMG_HEIGHT)
      - Convert to grayscale (if CHANNELS == 1)
      - Normalize to [0,1]
      - Add batch dimension
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Resize to 256x64 (W x H) to match (64, 256, 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

    if CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[..., np.newaxis]  # (H, W, 1)

    img = img.astype("float32") / 255.0

    # Add batch dimension: (1, 64, 256, 1)
    img = np.expand_dims(img, axis=0)
    return img


def ctc_greedy_decode(preds: np.ndarray) -> str:
    """
    Greedy decode for CTC-style output.
    preds shape: (batch, time_steps, num_classes)
    """
    # Take best class at each time step
    best_path = np.argmax(preds, axis=-1)[0]  # (time_steps,)

    decoded = []
    prev = -1
    for idx in best_path:
        if idx == BLANK_IDX:
            prev = -1
            continue
        if idx == prev:
            continue
        if idx < len(ALPHABET):
            decoded.append(ALPHABET[idx])
        prev = idx

    return "".join(decoded)


def predict_plate(img_path: str) -> str:
    """
    Full inference pipeline for one image:
      - load model
      - preprocess image
      - run predict
      - decode sequence to plate text
    """
    model = load_crnn_model()
    x = preprocess_image(img_path)
    preds = model.predict(x)  # (1, T, num_classes)
    plate_text = ctc_greedy_decode(preds)
    return plate_text


if __name__ == "__main__":
    # Ask user for image path (e.g., /home/roshan/Complate_ANPR/temp_crops/MH02BR7723.jpg)
    img_path = input("enter the img path :-").strip()

    if not os.path.isfile(img_path):
        print(f"[ERR] File not found: {img_path}")
    else:
        text = predict_plate(img_path)
        print("Predicted plate:", text)

