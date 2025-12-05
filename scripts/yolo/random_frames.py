import cv2
import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = "data/PXL_20250317_101139365.TS (1).mp4" 
OUTPUT_DIR = "data/dataset_yolo_50_frames"     
NUM_IMAGES_TO_SAVE = 50            # Extract 50 images this time

# ==========================================
# EXTRACTION LOGIC
# ==========================================
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames == 0:
    print("Error: Could not read video.")
    exit()

# Calculate stride to get exactly 50 images evenly spaced
stride = total_frames // NUM_IMAGES_TO_SAVE

print(f"Video has {total_frames} frames.")
print(f"Extracting 1 frame every {stride} frames...")

count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    if count % stride == 0:
        # Save with a distinct name
        filename = os.path.join(OUTPUT_DIR, f"batch2_img_{saved:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        saved += 1
        
        if saved >= NUM_IMAGES_TO_SAVE:
            break
            
    count += 1

cap.release()
print(f"\nDone! Saved {saved} new images to '{OUTPUT_DIR}'.")