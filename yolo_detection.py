import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.getcwd(), "Yolo_Module", "best.pt")

model = YOLO(MODEL_PATH)

def detect_and_crop(image_path: str) -> np.ndarray | None:

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERR] Cannot read image: {image_path}")
        return None

    h, w = img.shape[:2]

    results = model.predict(img, imgsz=640, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print(f"[WARN] No plate detected in: {image_path}")
        return None

    best = results.boxes[0]
    xmin, ymin, xmax, ymax = best.xyxy[0].cpu().numpy().astype(int)

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w - 1, xmax)
    ymax = min(h - 1, ymax)

    crop = img[ymin:ymax, xmin:xmax]
    return crop

def process_image(image_path: str, plate_text: str):
    crop = detect_and_crop(image_path)
    if crop is None:
        return None, None

    
    name = "".join(c for c in plate_text if c.isalnum())
    if not name:
        name = "unknown_plate"

    return crop, name

if __name__ == "__main__":

    '''test_image = "/home/roshan/Pictures/2.jpeg"             
    plate_number = input('enter the vehical number')          
    crop, name = process_image(test_image, plate_number)
    if crop is not None:
        cv2.imwrite(f"{name}.jpg", crop)
        print(f"Saved debug crop as {name}.jpg")'''
