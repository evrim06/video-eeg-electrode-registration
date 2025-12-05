import cv2
import os
from inference_sdk import InferenceHTTPClient

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = "C:\\Users\\zugo4834\\Desktop\\video-eeg-electrode-registration\\data\\dataset_yolo_50_frames"       # Folder containing your raw frames
OUTPUT_FOLDER = "inference_results"    # Folder to save images with boxes
API_KEY = "QuMPkya8M01yOJ7Ga0DT"               # Your Roboflow Private Key
MODEL_ID = "eeg-electrode-detection/2"     # Your Model ID (e.g. "my-project/1")
CONFIDENCE = 0.40                      # Filter out weak detections

# ==========================================
# SETUP
# ==========================================
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

print(f"Processing images from '{INPUT_FOLDER}'...")

# ==========================================
# PROCESSING LOOP
# ==========================================
for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
        
    image_path = os.path.join(INPUT_FOLDER, filename)
    save_path = os.path.join(OUTPUT_FOLDER, filename)
    
    # 1. Run Inference
    result = client.infer(image_path, model_id=MODEL_ID)
    
    # 2. Load Image for Drawing
    image = cv2.imread(image_path)
    
    # 3. Draw Boxes
    predictions = result['predictions']
    for p in predictions:
        if p['confidence'] < CONFIDENCE: continue
        
        # Roboflow returns Center-X, Center-Y, Width, Height
        x, y, w, h = p['x'], p['y'], p['width'], p['height']
        
        # Convert to Top-Left Corner for OpenCV
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Draw Rectangle (Green)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Label (Optional)
        label = f"{p['confidence']:.2f}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. Save Result
    cv2.imwrite(save_path, image)
    print(f"Saved: {save_path} ({len(predictions)} detections)")

print("\nDone! Open the 'inference_results' folder to inspect the boxes.")