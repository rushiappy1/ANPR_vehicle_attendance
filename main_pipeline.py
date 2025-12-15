# main_pipeline.py
from yolo_detection import process_image
from save_image import save_images_and_label
import os
import subprocess
import json

def run_pipeline(image_path: str, plate_text: str):
    """Full pipeline: YOLO → crop → save → CRNN inference → results"""
    
    # Step 1: YOLO detection
    print(f"[1/3] YOLO detecting plate in {image_path}")
    crop, base_name = process_image(image_path, plate_text)
    if crop is None:
        print("[ERR] Pipeline failed: no crop detected.")
        return None
    
    # Step 2: Save cropped plate
    print(f"[2/3] Saving crop as {base_name}")
    save_images_and_label(crop, base_name, plate_text)
    
    # Step 3: Run your existing CRNN inference on Temp/ folder
    print("[3/3] Running CRNN inference...")
    try:
        # Call your run_inference.py script
        result = subprocess.run(['python', 'run_inference.py'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"[ERR] CRNN inference failed: {result.stderr}")
            return None
        
        print(result.stdout)
        
        # Read the JSON results
        if os.path.exists("crnn_results.json"):
            with open("crnn_results.json", 'r') as f:
                crnn_results = json.load(f)
            
            # Add pipeline metadata
            final_result = {
                "input_image": os.path.basename(image_path),
                "expected_plate": plate_text,
                "yolo_crop_base_name": base_name,
                "crnn_predictions": crnn_results
            }
            
            # Save combined pipeline result
            pipeline_json = "pipeline_complete.json"
            with open(pipeline_json, 'w') as f:
                json.dump(final_result, f, indent=2)
            
            print(f"\nPipeline complete! Results:")
            print(f"  Input: {image_path}")
            print(f"  Expected: {plate_text}")
            print(f"  Crop: {base_name}")
            print(f"  CRNN predictions: {len(crnn_results)} plates")
            print(f"  Saved: {pipeline_json}")
            
            return final_result
            
    except Exception as e:
        print(f"[ERR] Failed to run inference: {e}")
        return None

if __name__ == "__main__":
    img_path = input('Enter the Img Path: ').strip()
    plate = input('Enter the Vehicle Number: ').strip()
    
    if not os.path.exists(img_path):
        print(f"[ERR] Image not found: {img_path}")
    else:
        run_pipeline(img_path, plate)
