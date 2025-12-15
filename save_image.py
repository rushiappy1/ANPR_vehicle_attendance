import os
import cv2
import numpy as np
import glob

TARGET_WIDTH = 700
TARGET_HEIGHT = 300
INTERP = cv2.INTER_CUBIC  

BASE_OUTPUT = os.path.join(os.getcwd(), "TrainingData")
TEMP_DIR = os.path.join("Temp")
CNDATA_DIR = os.path.join(BASE_OUTPUT, "CNN_Data_Training")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CNDATA_DIR, exist_ok=True)

def resize_plate(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    scale_w = TARGET_WIDTH / w
    scale_h = TARGET_HEIGHT / h
    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=INTERP)

    canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=img.dtype)

    x_offset = (TARGET_WIDTH - new_w) // 2
    y_offset = (TARGET_HEIGHT - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas

def _clear_temp_folder():
    for filename in os.listdir(TEMP_DIR):  
        file_path = os.path.join(TEMP_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)  


def save_images_and_label(crop_img: np.ndarray, base_name: str, plate_text: str):
   
    # Convert to grayscale and apply Otsu thresholding for B&W
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert back to 3-channel for consistency with resize_plate
    bw_3ch = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    
    resized = resize_plate(bw_3ch)

    img_filename = f"{base_name}.jpg"
    txt_filename = f"{base_name}.txt"

    temp_img_path = os.path.join(TEMP_DIR, img_filename)
    cndata_img_path = os.path.join(CNDATA_DIR, img_filename)
    cndata_txt_path = os.path.join(CNDATA_DIR, txt_filename)


    _clear_temp_folder()
    cv2.imwrite(temp_img_path, resized)

    cv2.imwrite(cndata_img_path, resized)

    with open(cndata_txt_path, "w", encoding="utf-8") as f:
        f.write(plate_text.strip())


    print(f"[OK] Saved temp image: {temp_img_path}")
    print(f"[OK] Saved cndata image: {cndata_img_path}")
    print(f"[OK] Saved label: {cndata_txt_path}")

if __name__ == "__main__":
    dummy = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
    save_images_and_label(dummy, "TESTPLATE1234", "TESTPLATE1234")
