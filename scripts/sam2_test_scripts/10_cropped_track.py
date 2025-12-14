import os
import sys
import cv2
import numpy as np
import torch
import sam2
import json
import copy
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import shutil
import pickle
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "video_path": "data/IMG_2763.mp4",
    "checkpoint_path": "checkpoints/sam2_hiera_large.pt",
    "model_cfg": "sam2_hiera_l.yaml",
    "frame_stride": 1,          
    "display_height": 800,
    "max_display_width": 1280,
    
    # FILTERS
    "min_area": 150,            
    "max_area": 3500,           
    "min_circularity": 0.60,    
    "min_solidity": 0.85,       
    "border_margin": 20,
    "duplicate_radius": 60      
}

FRAME_DIR = "temp_frames"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "tracking_data.pkl")
SMOOTH_FILE = os.path.join(RESULTS_DIR, "tracking_data_smoothed.pkl")

# ==========================================
# 1. SETUP
# ==========================================
print("--- STEP 1: INITIALIZATION ---")
if not os.path.exists(CONFIG["video_path"]):
    print(f"CRITICAL ERROR: Video not found at {CONFIG['video_path']}")
    sys.exit(1)

if os.path.exists(FRAME_DIR): shutil.rmtree(FRAME_DIR)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Models on {device}...")

sam2_dir = os.path.dirname(sam2.__file__)
yaml_path = os.path.join(sam2_dir, "configs", "sam2", CONFIG["model_cfg"])

sam2_model = build_sam2(yaml_path, CONFIG["checkpoint_path"], device=device)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model, points_per_side=64, 
    pred_iou_thresh=0.8, stability_score_thresh=0.9
)
predictor = build_sam2_video_predictor(yaml_path, CONFIG["checkpoint_path"], device=device)

# ==========================================
# 2. CROP SELECTION
# ==========================================
print("\n--- STEP 2: CROP SELECTION ---")
cap = cv2.VideoCapture(CONFIG["video_path"])
ret, first_frame = cap.read()
if not ret: 
    print("Error reading video.")
    sys.exit(1)

# Resize for display
orig_h, orig_w = first_frame.shape[:2]
scale_ratio = CONFIG["display_height"] / orig_h
disp_w = int(orig_w * scale_ratio)
disp_h = int(orig_h * scale_ratio)
small_frame = cv2.resize(first_frame, (disp_w, disp_h))

print(f"Original Size: {orig_w}x{orig_h}")
print(f"Display Size: {disp_w}x{disp_h} (Scale: {scale_ratio:.2f})")
print(">> PLEASE DRAW THE BOX NOW <<")

roi_small = cv2.selectROI("DRAW CROP BOX", small_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("DRAW CROP BOX")

# Calculate Upscaled Coordinates
sx = int(roi_small[0] / scale_ratio)
sy = int(roi_small[1] / scale_ratio)
sw = int(roi_small[2] / scale_ratio)
sh = int(roi_small[3] / scale_ratio)

# Validation
if sw < 50 or sh < 50:
    print(f"WARNING: Crop box is tiny or zero ({sw}x{sh}). Reverting to FULL FRAME.")
    sx, sy, sw, sh = 0, 0, orig_w, orig_h
else:
    # Ensure bounds
    sx = max(0, sx)
    sy = max(0, sy)
    sw = min(sw, orig_w - sx)
    sh = min(sh, orig_h - sy)

print(f"Final Crop Region: x={sx}, y={sy}, w={sw}, h={sh}")

# ==========================================
# 3. EXTRACTION LOOP
# ==========================================
print("\n--- STEP 3: EXTRACTING FRAMES ---")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_names = []
idx = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    if idx % CONFIG["frame_stride"] == 0:
        # Perform Crop
        cropped_frame = frame[sy:sy+sh, sx:sx+sw]
        
        # Validation: Is cropped frame empty?
        if cropped_frame.size == 0:
            print(f"Error: Frame {idx} crop resulted in empty image. Skipping.")
            continue
            
        fname = f"{saved_count:05d}.jpg"
        cv2.imwrite(os.path.join(FRAME_DIR, fname), cropped_frame)
        frame_names.append(fname)
        saved_count += 1
        
        if saved_count % 50 == 0:
            print(f"Saved {saved_count} frames...", end="\r")
            
    idx += 1
cap.release()
print(f"\nSuccessfully extracted {len(frame_names)} frames.")

if len(frame_names) == 0:
    print("CRITICAL ERROR: No frames saved. Exiting.")
    sys.exit(1)

# ==========================================
# 4. INITIALIZE TRACKER
# ==========================================
print("\n--- STEP 4: INITIALIZING AI (This takes time...) ---")
inference_state = predictor.init_state(video_path=FRAME_DIR)
print("AI Initialized.")

# ==========================================
# 5. MULTI-VIEW INTERFACE (DEBUGGED)
# ==========================================
print("\n--- STEP 5: STARTING GUI ---")
GLOBAL_REGISTERED_POINTS = [] 
current_id_counter = 0
current_idx = 0

# Helper Functions
def is_duplicate(new_pt, existing_points, radius):
    for old_pt in existing_points:
        if np.linalg.norm(np.array(new_pt) - np.array(old_pt)) < radius: return True
    return False

def get_robust_blobs(image_rgb):
    masks = mask_generator.generate(image_rgb)
    centroids = []
    h, w = image_rgb.shape[:2]
    for m in masks:
        area = m["area"]
        if area > 10000 or not (CONFIG["min_area"] < area < CONFIG["max_area"]): continue
        
        mask_uint = m["segmentation"].astype(np.uint8)
        cnts, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = cnts[0]
        
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if (bx < CONFIG["border_margin"]) or (by < CONFIG["border_margin"]) or \
           (bx + bw > w - CONFIG["border_margin"]) or (by + bh > h - CONFIG["border_margin"]): continue

        if cv2.arcLength(cnt, True) == 0: continue
        circ = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)
        if circ < CONFIG["min_circularity"]: continue

        hull = cv2.convexHull(cnt)
        if cv2.contourArea(hull) == 0: continue
        if area / float(cv2.contourArea(hull)) < CONFIG["min_solidity"]: continue

        M = cv2.moments(cnt)
        if M["m00"] != 0: centroids.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
    return centroids

def draw_hud(img, f_idx, tot_f, reg_c):
    overlay = img.copy()
    lines = [f"Frame: {f_idx}/{tot_f}", f"Registered: {reg_c}", "-"*15, "[S] Fwd", "[A] Back", "[D] Detect", "[Space] Done"]
    cv2.rectangle(overlay, (10, 10), (250, 10 + len(lines)*30), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (20, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return img

cv2.namedWindow("Guided Detector", cv2.WINDOW_AUTOSIZE)

print("Starting GUI Loop. If window is grey, click it and press 'S'.")

while True:
    img_path = os.path.join(FRAME_DIR, frame_names[current_idx])
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        break
    
    # Resize for Display
    h, w = img.shape[:2]
    scale = min(CONFIG["max_display_width"]/w, CONFIG["display_height"]/h)
    disp_img = cv2.resize(img, (int(w*scale), int(h*scale)))
    
    # Draw Points
    for pt in GLOBAL_REGISTERED_POINTS:
        cv2.circle(disp_img, (int(pt[0]*scale), int(pt[1]*scale)), 4, (0,0,255), -1)
    
    disp_img = draw_hud(disp_img, current_idx, len(frame_names), len(GLOBAL_REGISTERED_POINTS))
    
    cv2.imshow("Guided Detector", disp_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'): current_idx = min(current_idx + 15, len(frame_names)-1)
    elif key == ord('a'): current_idx = max(current_idx - 15, 0)
    elif key == ord('d'):
        print(f"Detecting on Frame {current_idx}...")
        cands = get_robust_blobs(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for pt in cands:
            if not is_duplicate(pt, GLOBAL_REGISTERED_POINTS, CONFIG["duplicate_radius"]):
                predictor.add_new_points_or_box(inference_state, frame_idx=current_idx, obj_id=current_id_counter, points=[np.array(pt, dtype=np.float32)], labels=[1])
                GLOBAL_REGISTERED_POINTS.append(pt)
                current_id_counter += 1
        print(f"Total Registered: {current_id_counter}")
        
    elif key == 32: break

cv2.destroyAllWindows()

# ==========================================
# 6. PROPAGATE & SAVE
# ==========================================
print(f"\n--- STEP 6: TRACKING ({len(GLOBAL_REGISTERED_POINTS)} objects) ---")
video_results = {}
try:
    with torch.inference_mode():
        for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(predictor.propagate_in_video(inference_state), total=len(frame_names)):
            frame_data = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    # Un-crop coordinates
                    frame_data[out_obj_id] = (np.mean(xs) + sx, np.mean(ys) + sy)
            video_results[out_frame_idx] = frame_data
            if out_frame_idx % 10 == 0:
                with open(RESULTS_FILE, "wb") as f: pickle.dump(video_results, f)
except KeyboardInterrupt:
    print("Stopped.")

with open(RESULTS_FILE, "wb") as f: pickle.dump(video_results, f)
print("Done.")