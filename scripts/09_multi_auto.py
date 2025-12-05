import os
import cv2
import numpy as np
import torch
import sam2
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import shutil
import pickle
from tqdm import tqdm
from scipy.signal import savgol_filter 

# ==========================================
# CONFIGURATION
# ==========================================
checkpoint_path = "checkpoints/sam2_hiera_large.pt"
video_path = "data/IMG_2763.mp4"
frame_dir = "temp_frames"
results_file = "results/tracking_data.pkl"

# STRATEGY: Full Quality (Fixes drifting)
FRAME_STRIDE = 1

# ROBUST FILTERS (Updated for 4K/1080p Resolution)
MIN_AREA = 100          # Min pixels 
MAX_AREA = 15000        # Increased for full resolution
MIN_CIRCULARITY = 0.65  
MIN_SOLIDITY = 0.85     
BORDER_MARGIN = 20      

# DUPLICATE PREVENTION 
DUPLICATE_RADIUS = 60   # Increased for full resolution

# ==========================================
# 1. SETUP MODELS
# ==========================================
sam2_dir = os.path.dirname(sam2.__file__)
model_cfg = os.path.join(sam2_dir, "configs", "sam2", "sam2_hiera_l.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Models on {device}...")

# --- FIX: Load Generator manually ---
# Step A: Load the base model weights
sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)

# Step B: Wrap it in the Generator Class
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=64,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.9
)

# Step C: Load Video Predictor (Separate instance)
predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)

# ==========================================
# 2. EXTRACT FRAMES (Full Quality)
# ==========================================
if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
os.makedirs(frame_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_names = []
idx = 0
saved_count = 0
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Extracting frames (Stride {FRAME_STRIDE} for Max Accuracy)...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    if idx % FRAME_STRIDE == 0:
        # FULL RESOLUTION (No Resize) to fix visibility issues
        fname = f"{saved_count:05d}.jpg"
        cv2.imwrite(os.path.join(frame_dir, fname), frame)
        frame_names.append(fname)
        saved_count += 1
    idx += 1
cap.release()

inference_state = predictor.init_state(video_path=frame_dir)

# ==========================================
# 3. SMART DETECTION LOGIC
# ==========================================
GLOBAL_REGISTERED_POINTS = [] 

def is_duplicate(new_pt, existing_points, radius):
    for old_pt in existing_points:
        dist = np.linalg.norm(np.array(new_pt) - np.array(old_pt))
        if dist < radius:
            return True
    return False

def get_robust_blobs(image_rgb):
    masks = mask_generator.generate(image_rgb)
    centroids = []
    
    h, w = image_rgb.shape[:2]
    
    for m in masks:
        mask = m["segmentation"]
        area = m["area"]
        if not (MIN_AREA < area < MAX_AREA): continue
        
        mask_uint = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = contours[0]
        
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if (x < BORDER_MARGIN) or (y < BORDER_MARGIN) or \
           (x + w_box > w - BORDER_MARGIN) or (y + h_box > h - BORDER_MARGIN):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY: continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area / float(hull_area)
        if solidity < MIN_SOLIDITY: continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
            
    return centroids

# ==========================================
# 4. MULTI-VIEW INTERFACE
# ==========================================
print("\n--- MULTI-VIEW DETECTION MODE ---")
print("1. 'A'/'S' to move.")
print("2. 'D' to DETECT (Duplicates ignored).")
print("3. 'Space' when done.")

current_id_counter = 0
current_idx = 0
cv2.namedWindow("Robust Detector")

while True:
    img_path = os.path.join(frame_dir, frame_names[current_idx])
    img = cv2.imread(img_path)
    display_img = img.copy()
    
    cv2.putText(display_img, f"Frame: {current_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_img, f"Registered: {len(GLOBAL_REGISTERED_POINTS)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    for pt in GLOBAL_REGISTERED_POINTS:
        cv2.circle(display_img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

    cv2.imshow("Robust Detector", display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'): 
        current_idx = min(current_idx + 5, len(frame_names) - 1)
    elif key == ord('a'): 
        current_idx = max(current_idx - 5, 0)
    
    elif key == ord('d'):
        print(f"\n--- Detecting on Frame {current_idx} ---")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        candidates = get_robust_blobs(img_rgb)
        print(f"SAM2 found {len(candidates)} blobs.")
        
        added = 0
        skipped = 0
        for pt in candidates:
            if is_duplicate(pt, GLOBAL_REGISTERED_POINTS, DUPLICATE_RADIUS):
                skipped += 1
                continue
            
            predictor.add_new_points_or_box(
                inference_state,
                frame_idx=current_idx,
                obj_id=current_id_counter,
                points=[np.array([pt[0], pt[1]], dtype=np.float32)],
                labels=[1]
            )
            GLOBAL_REGISTERED_POINTS.append(pt)
            current_id_counter += 1
            added += 1
            
        print(f"-> Added {added} NEW. Ignored {skipped} DUPLICATES.")
        cv2.putText(display_img, f"+{added} Added!", (vid_w//2-100, vid_h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow("Robust Detector", display_img)
        cv2.waitKey(500)

    elif key == 32: break

cv2.destroyAllWindows()

# ==========================================
# 5. PROPAGATE & SAVE
# ==========================================
print(f"\nTracking {len(GLOBAL_REGISTERED_POINTS)} objects...")
video_results = {}
os.makedirs("results", exist_ok=True)

try:
    with torch.inference_mode():
        for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(predictor.propagate_in_video(inference_state), total=len(frame_names)):
            frame_data = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = np.mean(xs), np.mean(ys)
                    frame_data[out_obj_id] = (cx, cy)
            video_results[out_frame_idx] = frame_data
            if out_frame_idx % 10 == 0:
                with open(results_file, "wb") as f: pickle.dump(video_results, f)

except KeyboardInterrupt:
    print("Stopped. Saving...")

with open(results_file, "wb") as f: pickle.dump(video_results, f)

# ==========================================
# 6. SMOOTHING
# ==========================================
print("\n--- Smoothing ---")
with open(results_file, "rb") as f: raw_data = pickle.load(f)

trajectories = {}
for frame_idx, objs in raw_data.items():
    for obj_id, coords in objs.items():
        if coords is None: continue
        if obj_id not in trajectories: trajectories[obj_id] = {"x": [], "y": [], "frames": []}
        trajectories[obj_id]["x"].append(coords[0])
        trajectories[obj_id]["y"].append(coords[1])
        trajectories[obj_id]["frames"].append(frame_idx)

smoothed_results = raw_data.copy()
smooth_window = 7

for obj_id, track in trajectories.items():
    if len(track["x"]) > smooth_window:
        smooth_x = savgol_filter(track["x"], window_length=smooth_window, polyorder=2)
        smooth_y = savgol_filter(track["y"], window_length=smooth_window, polyorder=2)
        for i, frame_idx in enumerate(track["frames"]):
            if obj_id in smoothed_results[frame_idx]:
                smoothed_results[frame_idx][obj_id] = (smooth_x[i], smooth_y[i])

with open("results/tracking_data_smoothed.pkl", "wb") as f:
    pickle.dump(smoothed_results, f)

print(f"DONE! Smoothed data saved to: results/tracking_data_smoothed.pkl")