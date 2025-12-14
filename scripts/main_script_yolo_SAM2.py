import sys
import os
import cv2
import numpy as np
import torch
import pickle
import json
import shutil
import gc
import time
import urllib.request
from tqdm import tqdm
from scipy.signal import savgol_filter
from ultralytics import YOLO
import sam2
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from contextlib import nullcontext

# ==================================================================================================
# 1. CONFIGURATION & GLOBAL SETTINGS
# ==================================================================================================

# --- Device & Precision Selection ---
# Automatically detect hardware and set the optimal precision
if torch.cuda.is_available():
    DEVICE_STR = "cuda"
    # Check if this specific GPU supports BFloat16 (e.g., Ampere A100/3090/4090 or newer)
    if torch.cuda.is_bf16_supported():
        USE_BFLOAT16 = True
        print("--- Using device: CUDA (BFloat16 Supported) ---")
        # Enable TF32 for faster math on supported cards
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        USE_BFLOAT16 = False
        print("--- Using device: CUDA (BFloat16 NOT Supported - Fallback to Float32) ---")
else:
    DEVICE_STR = "cpu"
    USE_BFLOAT16 = False
    print("--- Using device: CPU (Float32 Mode) ---")

# --- Dynamic Path Management ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

print(f"Project Root detected at: {BASE_DIR}")

# Define all file paths
VIDEO_PATH      = os.path.join(BASE_DIR, "data", "IMG_2763.mp4")
FRAME_DIR       = os.path.join(BASE_DIR, "frames")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")

RAW_FILE        = os.path.join(RESULTS_DIR, "tracking_raw.pkl")
SMOOTH_FILE     = os.path.join(RESULTS_DIR, "tracking_smoothed.pkl")
ORDER_FILE      = os.path.join(RESULTS_DIR, "electrode_order.json")
CROP_INFO_FILE  = os.path.join(RESULTS_DIR, "crop_info.json")

# --- YOLO Configuration ---
YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train4", "weights", "best.pt")

# --- SAM2 Model Configuration ---
SAM2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam2_hiera_small.pt")
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"

if not os.path.exists(SAM2_CHECKPOINT):
    print(f"Downloading SAM2 checkpoint to {SAM2_CHECKPOINT}...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, SAM2_CHECKPOINT)
else:
    print(f"SAM2 Checkpoint found at: {SAM2_CHECKPOINT}")

# --- SAM2 Config File Logic ---
SAM2_CONFIG_NAME = "sam2_hiera_s.yaml"
SAM2_CONFIG = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2", SAM2_CONFIG_NAME)

if not os.path.exists(SAM2_CONFIG):
    PROJECT_CONFIG = os.path.join(BASE_DIR, "configs", "sam2", SAM2_CONFIG_NAME)
    PROJECT_CONFIG_SIMPLE = os.path.join(BASE_DIR, "configs", SAM2_CONFIG_NAME)
    
    if os.path.exists(PROJECT_CONFIG):
        SAM2_CONFIG = PROJECT_CONFIG
    elif os.path.exists(PROJECT_CONFIG_SIMPLE):
        SAM2_CONFIG = PROJECT_CONFIG_SIMPLE
    else:
        print(f"Warning: Config not found at {SAM2_CONFIG} or in project configs.")
        SAM2_CONFIG = os.path.join(SCRIPT_DIR, SAM2_CONFIG_NAME)

print(f"Using SAM2 Config: {SAM2_CONFIG}")

# --- General Pipeline Settings ---
CONFIG = {
    "display_height": 800,      
    "duplicate_radius": 50,     
    "yolo_conf": 0.25,          
    "smooth_window": 7,         
    "poly_order": 2,            
    "cap_mask_expansion": 1.10,   
    "cap_min_area_frac": 0.03,    
    "cap_max_area_frac": 0.70,    
    "cap_confirm_alpha": 0.45,    
    "cap_autodetect_use_yolo": True, 
}

# ==================================================================================================
# 2. HELPER FUNCTIONS
# ==================================================================================================

def initialize_models(yolo_weights_path=YOLO_WEIGHTS, sam2_cfg_path=SAM2_CONFIG, sam2_ckpt_path=SAM2_CHECKPOINT, device_used=DEVICE_STR):
    """
    Loads YOLOv11s and SAM2.
    Ensures SAM2 model precision matches the global USE_BFLOAT16 setting.
    """
    print(f"Loading YOLO from {yolo_weights_path}...")
    if not os.path.exists(yolo_weights_path):
        print(f"ERROR: Custom YOLO weights not found at {yolo_weights_path}")
        sys.exit(1)
    yolo = YOLO(yolo_weights_path)

    print("Loading SAM2 Video Predictor...")
    sam2_predictor = build_sam2_video_predictor(sam2_cfg_path, sam2_ckpt_path, device=device_used)
    
    # --- PRECISION ALIGNMENT ---
    if USE_BFLOAT16:
        print("  -> Casting SAM2 model to bfloat16 (Speed Optimization)")
        sam2_predictor.model.to(dtype=torch.bfloat16)
    else:
        # On CPU or older GPUs, keep float32 (Default)
        pass
        
    return yolo, sam2_predictor

def resize_for_display(img, target_height):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_height))
    return resized, scale

def draw_hud(img, idx, total, current_id):
    msg1 = f"Frame: {idx}/{total-1}"
    status = []
    status.append(f"NAS: {'[Saved]' if current_id > 0 else '[Click]'}")
    status.append(f"LPA: {'[Saved]' if current_id > 1 else '[Click]'}")
    status.append(f"RPA: {'[Saved]' if current_id > 2 else '[Click]'}")

    if current_id <= 2:
        msg2 = " -> ".join(status)
        color_status = (0, 255, 255) 
    else:
        electrodes_done = current_id - 3
        msg2 = f"Landmarks: [Saved] | Electrodes Detected: {electrodes_done}"
        color_status = (0, 255, 0)   

    cv2.rectangle(img, (0, 0), (600, 100), (0, 0, 0), -1)
    cv2.putText(img, msg1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, msg2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
    cv2.putText(img, "Controls: s=fwd, a=back, d=detect, space=finish, q=quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img

def interactive_multiframe_crop(video_path, display_height):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path} for cropping.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_idx = 0
    roi = None
    drawing = False
    start_pt = None

    def on_mouse(event, x, y, flags, param):
        nonlocal roi, drawing, start_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            roi = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x0, y0 = start_pt
            roi = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Crop Preview")
    cv2.setMouseCallback("Crop Preview", on_mouse)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ret, frame = cap.read()
        if not ret: break

        disp, scale = resize_for_display(frame, display_height)
        show = disp.copy()

        cv2.rectangle(show, (0, 0), (show.shape[1], 120), (0, 0, 0), -1)
        cv2.putText(show, "CROP MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(show, "A / S : Move frames | SPACE: Confirm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        cv2.putText(show, "Draw ONE box containing cap in ALL frames", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow("Crop Preview", show)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('a'): current_idx = max(0, current_idx - 15) 
        elif key == ord('s'): current_idx = min(total_frames - 1, current_idx + 15) 
        elif key in (13, 32): 
            if roi is not None: break
        elif key == ord('q'): sys.exit(0)

    cv2.destroyWindow("Crop Preview")
    cap.release()

    x, y, w, h = roi
    sx = int(x / scale)
    sy = int(y / scale)
    sw = int(w / scale)
    sh = int(h / scale)
    return sx, sy, sw, sh

def extract_frames_with_crop():
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR) 
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n--- Interactive multi-frame cropping ---")
    try:
        sx, sy, sw, sh = interactive_multiframe_crop(VIDEO_PATH, CONFIG["display_height"])
    except Exception as e:
        print(f"Cropping failed or cancelled: {e}")
        sys.exit(1)

    if sw < 50 or sh < 50:
        print("Warning: Crop too small. Using full frame.")
        cap_tmp = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap_tmp.read()
        cap_tmp.release()
        sy, sx = 0, 0
        sh, sw = frame.shape[:2]

    offset_x, offset_y = sx, sy
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    FRAME_SKIP = 10 
    frames = []
    idx = 0
    frame_count = 0

    print(f"Extracting cropped frames (every {FRAME_SKIP}th frame)...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % FRAME_SKIP == 0:
            crop = frame[sy:sy + sh, sx:sx + sw]
            fname = f"{idx:05d}.jpg"
            cv2.imwrite(os.path.join(FRAME_DIR, fname), crop)
            frames.append(fname)
            idx += 1
        frame_count += 1
    cap.release()

    crop_info = {"x": sx, "y": sy, "w": sw, "h": sh, "skip": FRAME_SKIP}
    with open(CROP_INFO_FILE, "w") as f:
        json.dump(crop_info, f, indent=2)

    print(f"Extracted {len(frames)} cropped frames.")
    return frames, (offset_x, offset_y)

def order_electrodes_head_relative(smoothed_data):
    relative_positions = {}
    print(f"Calculating head-relative positions across {len(smoothed_data)} frames...")

    for frame_idx, objects in smoothed_data.items():
        if 0 not in objects or 1 not in objects or 2 not in objects: continue
        if objects[0] is None or objects[1] is None or objects[2] is None: continue

        NAS, LPA, RPA = np.array(objects[0]), np.array(objects[1]), np.array(objects[2])
        ear_center = (LPA + RPA) / 2.0
        
        vec_x = RPA - LPA
        norm_x = np.linalg.norm(vec_x)
        if norm_x == 0: continue
        unit_x = vec_x / norm_x
        
        vec_y = NAS - ear_center
        norm_y = np.linalg.norm(vec_y)
        if norm_y == 0: continue
        unit_y = vec_y / norm_y

        for obj_id, coords in objects.items():
            if obj_id < 3 or coords is None: continue
            P = np.array(coords)
            v = P - ear_center
            px = np.dot(v, unit_x) 
            py = np.dot(v, unit_y) 
            relative_positions.setdefault(obj_id, []).append([px, py])

    final_stats = []
    for obj_id, rel_list in relative_positions.items():
        if len(rel_list) < 5: continue 
        avg_rel = np.mean(rel_list, axis=0)
        final_stats.append({"id": obj_id, "rel_x": avg_rel[0], "rel_y": avg_rel[1]})

    sorted_electrodes = sorted(final_stats, key=lambda e: (-e["rel_y"], e["rel_x"]))
    return sorted_electrodes

def is_global_duplicate(candidate, existing_points, radius):
    for pt in existing_points[3:]: 
        if np.linalg.norm(np.array(candidate) - np.array(pt)) < radius:
            return True
    return False

# --- CAP MASK UTILS ---

def _resize_mask_to_disp(mask, disp_shape):
    return cv2.resize(mask.astype(np.uint8), (disp_shape[1], disp_shape[0]), interpolation=cv2.INTER_NEAREST) > 0

def _draw_mask_overlay(disp, cap_mask, alpha=0.45):
    out = disp.copy()
    cap = _resize_mask_to_disp(cap_mask, disp.shape)
    
    overlay = out.copy()
    overlay[cap == 0] = (overlay[cap == 0] * 0.35).astype(np.uint8)
    
    cap_cnts, _ = cv2.findContours(cap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cap_cnts, -1, (0, 255, 255), 2) 
    
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out

def _expand_mask(mask, expansion=1.10):
    if expansion <= 1.0: return mask.astype(bool)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return mask.astype(bool)
    
    h, w = (ys.max() - ys.min() + 1), (xs.max() - xs.min() + 1)
    base = max(h, w)
    
    k = max(3, int(base * (expansion - 1.0) * 0.5))
    if k % 2 == 0: k += 1 
    
    kernel = np.ones((k, k), np.uint8)
    return (cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0)

def _manual_click_cap_mask(img, sam2_predictor, state, frame_idx, display_h):
    disp, scale = resize_for_display(img, display_h)
    clicks = []
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x / scale), int(y / scale)))
    
    win = "CAP (manual): click center of cap | q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while not clicks:
        cv2.imshow(win, disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): sys.exit(0)
    cv2.destroyWindow(win)
    
    _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
        state, frame_idx=frame_idx, obj_id=999,
        points=[np.array(clicks[0], dtype=np.float32)], labels=[1]
    )
    cap_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
    return cap_mask

def _auto_cap_mask(img, sam2_cfg_path, sam2_ckpt_path, device_used, yolo=None, min_area_frac=0.03, max_area_frac=0.70, use_yolo=True):
    print("\n[Auto-Mask] Initializing SAM2 Automatic Mask Generator...")
    # NOTE: The auto-mask generator creates its own internal model.
    model = build_sam2(sam2_cfg_path, sam2_ckpt_path, device=device_used)
    
    # --- PRECISION ALIGNMENT ---
    # Must match the global setting derived from hardware support
    if USE_BFLOAT16:
        model.to(dtype=torch.bfloat16)

    amg = SAM2AutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.92, min_mask_region_area=500)
    
    print("[Auto-Mask] Generating masks...")
    
    # Use context manager ONLY if BFloat16 is enabled
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BFLOAT16 else nullcontext()
    
    with ctx:
        masks = amg.generate(img)
    
    del amg, model
    if device_used == "cuda": torch.cuda.empty_cache()
    gc.collect()

    H, W = img.shape[:2]
    area_min = H * W * float(min_area_frac)
    area_max = H * W * float(max_area_frac)

    centers = []
    if use_yolo and yolo is not None:
        res = yolo.predict(img, conf=0.20, verbose=False)
        if res and res[0].boxes is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            for b in boxes:
                centers.append(((b[0]+b[2])/2.0, (b[1]+b[3])/2.0))

    best, best_score = None, -1e9
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        area = float(seg.sum())
        if not (area_min <= area <= area_max): continue
        score = 0.0
        if centers:
            inside = 0
            for (cx, cy) in centers:
                if 0 <= int(cy) < H and 0 <= int(cx) < W and seg[int(cy), int(cx)] > 0:
                    inside += 1
            score += inside * 1000.0 
        score += area 
        if score > best_score:
            best_score = score
            best = seg.astype(bool)

    if best is None and masks: best = max(masks, key=lambda x: x["area"])["segmentation"].astype(bool)
    if best is None: best = np.ones((H, W), dtype=bool)
    return best

def create_cap_mask_all_features(first_img, sam2_predictor, state, frame_idx, sam2_cfg_path, sam2_ckpt_path, device_used, yolo=None):
    mode = "auto"
    cap_mask = None

    while True:
        if mode == "auto":
            cap_mask = _auto_cap_mask(first_img, sam2_cfg_path, sam2_ckpt_path, device_used, yolo, CONFIG["cap_min_area_frac"], CONFIG["cap_max_area_frac"], CONFIG["cap_autodetect_use_yolo"])
        else:
            cap_mask = _manual_click_cap_mask(first_img, sam2_predictor, state, frame_idx, CONFIG["display_height"])

        cap_mask_exp = _expand_mask(cap_mask, expansion=CONFIG["cap_mask_expansion"])
        disp, _ = resize_for_display(first_img, CONFIG["display_height"])
        preview = _draw_mask_overlay(disp, cap_mask_exp, alpha=CONFIG["cap_confirm_alpha"])
        
        instructions = f"CAP MASK (mode={mode}) | y=accept | r=redo | m=manual | a=auto | q=quit"
        tip = "Tip: For better detection, press 'm' and click the center of the cap."

        win = "Confirm Cap Mask"
        cv2.namedWindow(win)
        while True:
            show = preview.copy()
            cv2.rectangle(show, (0, 0), (show.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(show, instructions, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(show, tip, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
            
            cv2.imshow(win, show)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('y'):
                sam2_predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=999, mask=cap_mask_exp.astype(np.uint8))
                cv2.destroyWindow(win)
                return cap_mask_exp
            if k == ord('r'): break
            if k == ord('m'): mode = "manual"; break
            if k == ord('a'): mode = "auto"; break
            if k == ord('q'): sys.exit(0)
        cv2.destroyWindow(win)

def precompute_cap_masks(sam2_predictor, state, frame_names, device_str):
    print("Pre-tracking cap mask for all frames...")
    cap_masks = {}
    
    # --- Autocast Context ---
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BFLOAT16 else nullcontext()
    
    with ctx, torch.inference_mode():
        for f_idx, ids, logits in tqdm(sam2_predictor.propagate_in_video(state), total=len(frame_names)):
            if 999 in ids:
                idx = list(ids).index(999)
                mask = (logits[idx] > 0.0).cpu().numpy().squeeze()
                cap_masks[f_idx] = _expand_mask(mask, CONFIG["cap_mask_expansion"])
            else:
                cap_masks[f_idx] = None 
    return cap_masks

# ==================================================================================================
# 3. MAIN PIPELINE LOGIC
# ==================================================================================================

def main():
    # 1. Load Models (YOLOv11s + SAM2 Small)
    yolo, sam2_predictor = initialize_models()
    
    # 2. Extract Frames
    frame_names, (crop_off_x, crop_off_y) = extract_frames_with_crop()
    if not frame_names: return

    # 3. Init State for Main Tracking
    state = sam2_predictor.init_state(video_path=FRAME_DIR)
    
    # 4. Create Initial Cap Mask
    first_img = cv2.imread(os.path.join(FRAME_DIR, frame_names[0]))
    valid_mask = create_cap_mask_all_features(first_img, sam2_predictor, state, 0, SAM2_CONFIG, SAM2_CHECKPOINT, DEVICE_STR, yolo)
    
    # 5. Pre-compute Cap Masks (Optimization)
    print("Initializing temporary SAM2 state for cap tracking...")
    cap_state = sam2_predictor.init_state(video_path=FRAME_DIR)
    sam2_predictor.add_new_mask(cap_state, frame_idx=0, obj_id=999, mask=valid_mask.astype(np.uint8))
    cap_masks_cache = precompute_cap_masks(sam2_predictor, cap_state, frame_names, DEVICE_STR)
    
    # Free memory from temporary state
    del cap_state
    if DEVICE_STR == "cuda": torch.cuda.empty_cache()
    gc.collect()

    # 6. Interactive Labeling Phase
    disp0, DISPLAY_SCALE = resize_for_display(first_img, CONFIG["display_height"])
    global_points = []
    current_id, current_idx = 0, 0
    flash_points = []
    
    def on_click(event, x, y, flags, param):
        nonlocal current_id
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = int(x / DISPLAY_SCALE)
            real_y = int(y / DISPLAY_SCALE)
            
            # --- Check: Is click inside the Cap Mask? ---
            current_cap_mask = cap_masks_cache.get(current_idx)
            is_valid = True
            if current_id >= 3: # Enforce for electrodes
                if current_cap_mask is None: is_valid = False
                else:
                    h, w = current_cap_mask.shape
                    if not (0 <= real_y < h and 0 <= real_x < w and current_cap_mask[real_y, real_x]):
                        is_valid = False
            
            if not is_valid:
                print(">>> CLICK REJECTED: Outside Cap Mask")
                return

            sam2_predictor.add_new_points_or_box(state, frame_idx=current_idx, obj_id=current_id, points=[np.array((real_x, real_y), dtype=np.float32)], labels=[1])
            global_points.append((real_x, real_y))
            flash_points.append((real_x, real_y, (0, 255, 0) if current_id < 3 else (0, 0, 255), time.time()))
            current_id += 1

    cv2.namedWindow("Pipeline")
    cv2.setMouseCallback("Pipeline", on_click)

    print("\n--- Interactive Phase ---")
    while True:
        img = cv2.imread(os.path.join(FRAME_DIR, frame_names[current_idx]))
        if img is None: break
        disp, _ = resize_for_display(img, CONFIG["display_height"])
        
        # Visualize Cap Mask
        current_cap_mask = cap_masks_cache.get(current_idx)
        if current_cap_mask is not None:
            mask_overlay = disp.copy()
            resized_mask = cv2.resize(current_cap_mask.astype(np.uint8), (disp.shape[1], disp.shape[0]))
            mask_overlay[resized_mask == 0] = mask_overlay[resized_mask == 0] // 2
            disp = cv2.addWeighted(disp, 0.7, mask_overlay, 0.3, 0)

        # Visualize Clicks
        now = time.time()
        flash_points[:] = [p for p in flash_points if now - p[3] < 1.0]
        for (fx, fy, col, _) in flash_points:
            cv2.circle(disp, (int(fx * DISPLAY_SCALE), int(fy * DISPLAY_SCALE)), 7, col, -1)

        disp = draw_hud(disp, current_idx, len(frame_names), current_id)
        cv2.imshow("Pipeline", disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): current_idx = min(current_idx + 15, len(frame_names)-1)
        elif key == ord('a'): current_idx = max(current_idx - 15, 0)
        elif key == ord('q'): sys.exit(0)
        
        # YOLO Detection
        elif key == ord('d'):
            if current_id < 3: print(">>> Click landmarks first!"); continue
            print(f"YOLO frame {current_idx}...")
            results = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
            found = 0
            if results and results[0].boxes is not None:
                current_cap_mask = cap_masks_cache.get(current_idx)
                for box in results[0].boxes.xyxy.cpu().numpy():
                    cx, cy = (box[0] + box[2])/2.0, (box[1] + box[3])/2.0
                    # Check Mask
                    if current_cap_mask is not None:
                        if not (0 <= int(cy) < current_cap_mask.shape[0] and 0 <= int(cx) < current_cap_mask.shape[1] and current_cap_mask[int(cy), int(cx)]):
                            continue
                    # Check Duplicate
                    if is_global_duplicate((cx, cy), global_points, CONFIG["duplicate_radius"]): continue
                    # Add
                    sam2_predictor.add_new_points_or_box(state, frame_idx=current_idx, obj_id=current_id, box=box)
                    global_points.append((cx, cy))
                    flash_points.append((cx, cy, (0,0,255), time.time()))
                    current_id += 1; found += 1
            print(f"YOLO added {found} electrodes.")
        elif key == 32: 
            if current_id < 3: print(">>> Click landmarks first."); continue
            else: break
    cv2.destroyAllWindows()

    # 7. Final Tracking
    print("\n--- SAM2 Tracking ---")
    tracking = {}
    
    # --- Autocast Context ---
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BFLOAT16 else nullcontext()
    
    with ctx, torch.inference_mode():
        for f_idx, ids, logits in tqdm(sam2_predictor.propagate_in_video(state), total=len(frame_names)):
            frame_dict = {}
            for i, obj_id in enumerate(ids):
                if obj_id == 999: continue # Ignore Cap Mask in output
                mask = (logits[i] > 0.0).cpu().numpy().squeeze()
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = np.mean(xs) + crop_off_x, np.mean(ys) + crop_off_y
                    frame_dict[int(obj_id)] = (float(cx), float(cy))
            tracking[int(f_idx)] = frame_dict

    with open(RAW_FILE, "wb") as f: pickle.dump(tracking, f)

    # 8. Smoothing & Ordering
    smoothed = {}
    for f_idx, objs in tracking.items(): smoothed[f_idx] = dict(objs)
    traj = {}
    for f_idx, objs in tracking.items():
        for obj_id, (x, y) in objs.items():
            traj.setdefault(obj_id, {"x": [], "y": [], "frames": []})
            traj[obj_id]["frames"].append(f_idx)
            traj[obj_id]["x"].append(x)
            traj[obj_id]["y"].append(y)
    
    for obj_id, t in traj.items():
        if len(t["x"]) >= CONFIG["smooth_window"]:
            try:
                sx = savgol_filter(t["x"], CONFIG["smooth_window"], CONFIG["poly_order"])
                sy = savgol_filter(t["y"], CONFIG["smooth_window"], CONFIG["poly_order"])
                for k, f_idx in enumerate(t["frames"]):
                    smoothed[f_idx][obj_id] = (float(sx[k]), float(sy[k]))
            except: pass
    
    with open(SMOOTH_FILE, "wb") as f: pickle.dump(smoothed, f)
    ordered = order_electrodes_head_relative(smoothed)
    if ordered:
        with open(ORDER_FILE, "w") as f: json.dump([e["id"] for e in ordered], f)
        print(f"Saved order to {ORDER_FILE}")
    print("Done.")

if __name__ == "__main__":
    main()