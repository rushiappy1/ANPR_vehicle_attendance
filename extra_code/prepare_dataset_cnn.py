import os
import shutil

def rename_images_using_txt(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Group txt files by label
    label_to_txts = {}
    for file in os.listdir(source_dir):
        if file.endswith(".txt"):
            txt_path = os.path.join(source_dir, file)
            with open(txt_path, "r") as f:
                label = f.read().strip()
            if label:
                if label not in label_to_txts:
                    label_to_txts[label] = []
                label_to_txts[label].append(file)

    for label, txt_files in label_to_txts.items():
        num_files = len(txt_files)
        print(f"Label '{label}' has {num_files} txt files")

        for i, txt_file in enumerate(sorted(txt_files), 1):
            base_name = os.path.splitext(txt_file)[0]
            
            # Try .jpg first, then .jpeg, then .png
            img_extensions = ['.jpg', '.jpeg', '.png']
            img_path = None
            for ext in img_extensions:
                candidate = os.path.join(source_dir, base_name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            
            if not img_path:
                print(f"[MISSING] Image not found for {txt_file}")
                continue

            # Assign extension based on count and position
            if num_files == 1:
                new_ext = '.jpg'
            elif num_files == 2:
                new_ext = '.jpeg' if i == 2 else '.jpg'
            else:  # 3 or more
                if i == 1:
                    new_ext = '.jpg'
                elif i == 2:
                    new_ext = '.jpeg'
                elif i == 3:
                    new_ext = '.png'
                else:
                    new_ext = f'_{i}.jpg'  # extras get numbered

            new_img_name = f"{label}{new_ext}"
            output_path = os.path.join(output_dir, new_img_name)
            
            # Handle duplicate names by adding counter
            counter = 1
            original_output = output_path
            while os.path.exists(output_path):
                name, ext = os.path.splitext(original_output)
                output_path = f"{name}_{counter}{ext}"
                counter += 1
            
            shutil.copy(img_path, output_path)
            print(f"[OK] {os.path.basename(img_path)} → {os.path.basename(output_path)}")

# --------- RUN ----------
source_folder = os.path.join(os.getcwd(), "CRNNTrainingData")      # <-- CHANGE THIS
output_folder = os.path.join(os.getcwd(), "CNN_TrainingData")      # <-- CHANGE THIS

rename_images_using_txt(source_folder, output_folder)

