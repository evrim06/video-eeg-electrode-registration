import os
import sys
import cv2
import numpy as np
import torch
import json
import pickle
from tqdm import tqdm
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # PATHS
    "video_path": "data/IMG_2763.mp4",
    # POINT THIS TO YOUR TRAINED MODEL
    "yolo_model": r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\runs\detect\train\weights\best_v2.pt", 
    "sam2_checkpoint": "checkpoints/sam2_hiera_large.pt",
    "sam2_config": "configs/sam2/sam2_hiera_l.yaml",

    "max_display_width": 1280,  # Max window width
    
    # SETTINGS
    "frame_stride": 1,          # Process every frame (High Accuracy)
    "display_height": 850,      # Window size (Fits on laptop)
    "yolo_conf": 0.35,          # Confidence (From your training graphs)
    "duplicate_radius": 50      # Pixel distance to ignore duplicate clicks
}

# OUTPUT PATHS
FRAME_DIR = "temp_frames"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "tracking_data.pkl")

# ==========================================
# 1. SETUP MODELS
# ==========================================
# Check if model exists
if not os.path.exists(CONFIG["yolo_model"]):
    print(f"CRITICAL ERROR: YOLO model not found at {CONFIG['yolo_model']}")
    print("Please check the path to your best.pt file.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Models on {device}...")

# Load YOLO (The "Eye")
print(f"Loading YOLO: {CONFIG['yolo_model']}")
yolo = YOLO(CONFIG["yolo_model"])

# Load SAM2 (The "Tracker")
print("Loading SAM2...")
sam2_predictor = build_sam2_video_predictor(CONFIG["sam2_config"], CONFIG["sam2_checkpoint"], device=device)

# ==========================================
# 2. EXTRACT & CROP
# ==========================================
if os.path.exists(FRAME_DIR): shutil.rmtree(FRAME_DIR)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

cap = cv2.VideoCapture(CONFIG["video_path"])
ret, first_frame = cap.read()
if not ret: sys.exit("Error reading video")

# --- RESIZE FOR CROP SELECTION ---
# We resize the display so the window fits on your screen
orig_h, orig_w = first_frame.shape[:2]
scale_ratio = CONFIG["display_height"] / orig_h
disp_w = int(orig_w * scale_ratio)
disp_h = int(orig_h * scale_ratio)
small_frame = cv2.resize(first_frame, (disp_w, disp_h))

print("\n--- CROP STEP ---")
print("1. Draw a box around the HEAD (ignore the background).")
print("2. Press ENTER or SPACE.")
roi_small = cv2.selectROI("CROP SELECTION", small_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("CROP SELECTION")

# Upscale crop coordinates back to full resolution
sx = int(roi_small[0] / scale_ratio)
sy = int(roi_small[1] / scale_ratio)
sw = int(roi_small[2] / scale_ratio)
sh = int(roi_small[3] / scale_ratio)

if sw < 50: 
    print("No crop selected. Using full video.")
    sx, sy, sw, sh = 0, 0, orig_w, orig_h

print(f"Cropping to: x={sx}, y={sy}, w={sw}, h={sh}")

# --- EXTRACTION LOOP ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_names = []
idx = 0
saved_count = 0

print("Extracting frames...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if idx % CONFIG["frame_stride"] == 0:
        # Crop the high-res frame
        cropped = frame[sy:sy+sh, sx:sx+sw]
        fname = f"{saved_count:05d}.jpg"
        cv2.imwrite(os.path.join(FRAME_DIR, fname), cropped)
        frame_names.append(fname)
        saved_count += 1
    idx += 1
cap.release()

# Initialize SAM2 with the cropped frames
inference_state = sam2_predictor.init_state(video_path=FRAME_DIR)

# ==========================================
# 3. HYBRID INTERFACE
# ==========================================
GLOBAL_POINTS = []
current_id = 0 # 0,1,2 = Landmarks. 3+ = Electrodes
current_idx = 0

def is_duplicate(new_pt, existing_points, radius):
    for old_pt in existing_points:
        if np.linalg.norm(np.array(new_pt) - np.array(old_pt)) < radius: return True
    return False

def draw_hud(img, frame_idx):
    overlay = img.copy()
    lines = [
        f"Frame: {frame_idx}/{len(frame_names)}",
        f"Landmarks: {min(current_id, 3)}/3",
        f"Electrodes: {max(0, current_id-3)}",
        "-"*20,
        "1. CLICK Landmarks (Green Stickers)",
        "2. PRESS 'D' -> YOLO Auto-Detect",
        "3. 'S' / 'A' -> Move Video",
        "4. SPACE -> Finish & Track"
    ]
    cv2.rectangle(overlay, (5, 5), (450, 5 + len(lines)*35), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, line in enumerate(lines):
        col = (0, 255, 0) if "Landmarks" in line and current_id < 3 else (255, 255, 255)
        cv2.putText(img, line, (15, 35 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return img

def mouse_click(event, x, y, flags, param):
    global current_id
    # We map the display click back to the image coordinates
    # (Since we are viewing a resized version, but logic happens on full size)
    # Actually, to keep it simple, we will display at 1:1 if possible or handle scaling carefully.
    # For this script, we will trust the display is resized in the loop.
    # IMPORTANT: The mouse event gives coordinates relative to the WINDOW.
    # We must descale them inside the loop logic or simply ensure window matches image.
    pass 

# Note: Handling mouse clicks on a resized window is tricky. 
# SIMPLIFICATION: We will rely on 'D' for electrodes. 
# For Landmarks, we will just use the loop logic below.

cv2.namedWindow("Hybrid Pipeline", cv2.WINDOW_AUTOSIZE)

while True:
    img_path = os.path.join(FRAME_DIR, frame_names[current_idx])
    img = cv2.imread(img_path)
    if img is None: break
    
    # Resize for Display
    h, w = img.shape[:2]
    scale = min(CONFIG["max_display_width"]/w, CONFIG["display_height"]/h)
    disp_w, disp_h = int(w*scale), int(h*scale)
    disp_img = cv2.resize(img, (disp_w, disp_h))
    
    # Draw Points
    for i, pt in enumerate(GLOBAL_POINTS):
        color = (0, 255, 0) if i < 3 else (0, 0, 255)
        cv2.circle(disp_img, (int(pt[0]*scale), int(pt[1]*scale)), 5, color, -1)

    disp_img = draw_hud(disp_img, current_idx)
    cv2.imshow("Hybrid Pipeline", disp_img)
    
    # Handle Input
    key = cv2.waitKey(1)
    
    # Mouse Handling (Manual Check for Click)
    # We use cv2.setMouseCallback to capture the click, then process it here
    # This is a cleaner way to handle state
    def local_mouse(event, x, y, flags, param):
        global current_id
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale UP to real image size
            real_x = int(x / scale)
            real_y = int(y / scale)
            
            sam2_predictor.add_new_points_or_box(
                inference_state,
                frame_idx=current_idx,
                obj_id=current_id,
                points=[np.array([real_x, real_y], dtype=np.float32)],
                labels=[1]
            )
            GLOBAL_POINTS.append((real_x, real_y))
            print(f"Manual Click (ID {current_id}) at {real_x}, {real_y}")
            current_id += 1

    cv2.setMouseCallback("Hybrid Pipeline", local_mouse)

    if key & 0xFF == ord('s'): 
        current_idx = min(current_idx + 15, len(frame_names)-1)
    elif key & 0xFF == ord('a'): 
        current_idx = max(current_idx - 15, 0)
    
    # YOLO DETECT
    elif key & 0xFF == ord('d'):
        print(f"\n--- YOLO Detecting on Frame {current_idx} ---")
        # Run YOLO on FULL resolution image (not the display one)
        results = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        added = 0
        for box in boxes:
            # Check center for duplicate
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            if is_duplicate((cx, cy), GLOBAL_POINTS, CONFIG["duplicate_radius"]): continue
            
            # Pass Box to SAM2
            sam2_predictor.add_new_points_or_box(
                inference_state,
                frame_idx=current_idx,
                obj_id=current_id,
                box=box
            )
            GLOBAL_POINTS.append((cx, cy))
            current_id += 1
            added += 1
        print(f"YOLO found {added} new electrodes!")
        
    elif key & 0xFF == 32: # SPACE
        break

cv2.destroyAllWindows()

# ==========================================
# 4. TRACK & SAVE
# ==========================================
print(f"\nTracking {len(GLOBAL_POINTS)} objects...")
video_results = {}

try:
    with torch.inference_mode():
        for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(sam2_predictor.propagate_in_video(inference_state), total=len(frame_names)):
            frame_data = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = np.mean(xs), np.mean(ys)
                    # UN-CROP: Add offset back to get Real World coordinates
                    real_x = cx + sx
                    real_y = cy + sy
                    frame_data[out_obj_id] = (real_x, real_y)
            video_results[out_frame_idx] = frame_data
            
            # Auto-save
            if out_frame_idx % 10 == 0:
                with open(RESULTS_FILE, "wb") as f: pickle.dump(video_results, f)

except KeyboardInterrupt:
    print("\nStopped. Saving...")

with open(RESULTS_FILE, "wb") as f: pickle.dump(video_results, f)
print(f"Done! Results saved to {RESULTS_FILE}")