import cv2
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

# --- CONFIGURATION ---
# UPDATE THESE PATHS!
BASE_DIR = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration"
VIDEO_PATH = os.path.join(BASE_DIR, "data", "IMG_2763.mp4")
TRACKING_FILE = os.path.join(BASE_DIR, "results", "tracking_smoothed.pkl")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "results", "final_tracking_overlay.mp4")
CROP_INFO_FILE = os.path.join(BASE_DIR, "results", "crop_info.json")

# --- LOAD DATA ---
if not os.path.exists(TRACKING_FILE):
    print("Error: Tracking file not found.")
    sys.exit(1)

with open(TRACKING_FILE, "rb") as f:
    tracking_data = pickle.load(f)

# Load Crop Info (to align dots with original video)
# If crop info is missing, we assume no crop (offset 0,0)
try:
    import json
    with open(CROP_INFO_FILE, "r") as f:
        crop_info = json.load(f)
        OFFSET_X = crop_info.get("x", 0)
        OFFSET_Y = crop_info.get("y", 0)
        FRAME_SKIP = crop_info.get("skip", 1)
        print(f"Loaded Crop Info: Offset ({OFFSET_X}, {OFFSET_Y}), Skip {FRAME_SKIP}")
except:
    print("Warning: Crop info not found. Assuming full frame (0,0).")
    OFFSET_X, OFFSET_Y, FRAME_SKIP = 0, 0, 1

# --- SETUP VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps / FRAME_SKIP, (width, height))

print(f"Rendering video to: {OUTPUT_VIDEO}")
print(f"Total source frames: {total_frames}")

# --- PROCESS FRAMES ---
# We iterate through the TRACKED frames (which might be skipped)
sorted_tracked_frames = sorted(tracking_data.keys())

# Create a map of colors for each ID so they stay distinct
np.random.seed(42)
colors = {}

current_source_frame = 0
processed_count = 0

pbar = tqdm(total=len(sorted_tracked_frames))

for frame_idx in sorted_tracked_frames:
    # 1. Fast Forward to the correct frame in the video
    # Because we extracted every Nth frame, we must calculate the real frame number
    # If your tracking keys match the crop sequence (0, 1, 2...), we need to math it.
    
    # Simple logic: The tracking keys (0, 1, 2...) correspond to the *extracted* frames.
    # So tracking frame 0 is video frame 0.
    # Tracking frame 1 is video frame (1 * FRAME_SKIP).
    
    target_video_frame = frame_idx * FRAME_SKIP
    
    # Set video reader to that frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_video_frame)
    ret, frame = cap.read()
    if not ret: break

    # 2. Draw Electrodes
    objs = tracking_data[frame_idx]
    
    # Draw Landmarks (ID 0, 1, 2)
    landmarks = [0, 1, 2]
    landmark_labels = ["NAS", "LPA", "RPA"]
    
    for i, lid in enumerate(landmarks):
        if lid in objs and objs[lid] is not None:
            # Coordinates in the cropped frame
            cx, cy = objs[lid]
            
            # If the tracking data was saved RELATIVE to the crop, we add the offset.
            # If your pipeline saved ABSOLUTE coordinates (which it usually does if configured right),
            # we don't add the offset. 
            # *CHECK*: In the main script, we did `cx + off_x`. So data is GLOBAL.
            # We treat (cx, cy) as global coordinates directly.
            
            gx, gy = int(cx), int(cy)
            
            cv2.circle(frame, (gx, gy), 8, (0, 0, 255), -1) # Red for Landmarks
            cv2.putText(frame, landmark_labels[i], (gx+10, gy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw Electrodes (ID 3+)
    for obj_id, coords in objs.items():
        if obj_id < 3 or coords is None: continue
        
        # Color generation
        if obj_id not in colors:
            colors[obj_id] = np.random.randint(0, 255, 3).tolist()
        
        gx, gy = int(coords[0]), int(coords[1])
        
        # Draw dot
        cv2.circle(frame, (gx, gy), 5, colors[obj_id], -1)
        # Draw label
        cv2.putText(frame, str(obj_id), (gx+8, gy-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 3. Write Frame
    out.write(frame)
    pbar.update(1)

pbar.close()
cap.release()
out.release()

print("\nDone! Open the video to verify tracking.")
print(f"File: {OUTPUT_VIDEO}")