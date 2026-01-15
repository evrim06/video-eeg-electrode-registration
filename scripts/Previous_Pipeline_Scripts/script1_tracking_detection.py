"""

SCRIPT 1: ELECTRODE DETECTION & TRACKING


PURPOSE:
    Detect and track EEG electrodes + landmarks in 2D video frames.
    
INPUT:
    - Video file of head with EEG cap rotating
    
OUTPUT:
    - frames/                  : Extracted video frames
    - tracking_results.pkl     : {frame_idx: {object_id: (x, y), ...}, ...}
    - crop_info.json          : Crop region metadata

IMPORTANT:
    Use DIFFERENT colored stickers for landmarks:
    - NAS (Nasion): RED sticker
    - LPA (Left ear): BLUE sticker  
    - RPA (Right ear): GREEN sticker
    
    SAM2 cannot distinguish identical objects!

CONTROLS:
    Navigation: A = Back 15 frames, S = Forward 15 frames
    Actions:    D = YOLO Detect, R = Refresh map, Click = Add point
    Finish:     SPACE = Done, Q = Quit
"""

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
from ultralytics import YOLO


# SAM2 DTYPE PATCH


print("=" * 70)
print("SCRIPT 1: ELECTRODE DETECTION & TRACKING")
print("=" * 70)

torch.set_default_dtype(torch.float32)

import sam2
from sam2.modeling.memory_attention import MemoryAttentionLayer, MemoryAttention
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Store originals
_orig_rope = RoPEAttention.forward
_orig_mem_layer = MemoryAttentionLayer.forward
_orig_mem = MemoryAttention.forward

def _convert_dtype(obj, dtype):
    if obj is None: return None
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype) if obj.is_floating_point() and obj.dtype != dtype else obj
    if isinstance(obj, list): return [_convert_dtype(x, dtype) for x in obj]
    if isinstance(obj, tuple): return tuple(_convert_dtype(x, dtype) for x in obj)
    if isinstance(obj, dict): return {k: _convert_dtype(v, dtype) for k, v in obj.items()}
    return obj

def _get_dtype(m):
    for p in m.parameters():
        if p.is_floating_point(): return p.dtype
    return torch.float32

def _patched_rope(self, q, k, v, num_k_exclude_rope=0):
    with torch.cuda.amp.autocast(enabled=False):
        d = _get_dtype(self)
        return _orig_rope(self, _convert_dtype(q,d), _convert_dtype(k,d), _convert_dtype(v,d), num_k_exclude_rope)

def _patched_mem_layer(self, tgt, memory, pos=None, query_pos=None, num_k_exclude_rope=0):
    with torch.cuda.amp.autocast(enabled=False):
        d = _get_dtype(self)
        return _orig_mem_layer(self, _convert_dtype(tgt,d), _convert_dtype(memory,d), 
                               _convert_dtype(pos,d), _convert_dtype(query_pos,d), num_k_exclude_rope)

def _patched_mem(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
    with torch.cuda.amp.autocast(enabled=False):
        d = _get_dtype(self)
        return _orig_mem(self, _convert_dtype(curr,d), _convert_dtype(memory,d),
                        _convert_dtype(curr_pos,d), _convert_dtype(memory_pos,d), num_obj_ptr_tokens)

RoPEAttention.forward = _patched_rope
MemoryAttentionLayer.forward = _patched_mem_layer
MemoryAttention.forward = _patched_mem

print("SAM2 dtype patch applied")


# CONFIGURATION


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
print(f"Device: {DEVICE.upper()}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

VIDEO_PATH = os.path.join(BASE_DIR, "data", "IMG_2763.mp4")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
CROP_INFO_FILE = os.path.join(RESULTS_DIR, "crop_info.json")

YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train4", "weights", "best.pt")
SAM2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam2_hiera_small.pt")
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
SAM2_CONFIG_NAME = "sam2_hiera_s.yaml"

SAM2_CONFIG = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2", SAM2_CONFIG_NAME)
if not os.path.exists(SAM2_CONFIG):
    for p in [os.path.join(BASE_DIR, "configs", "sam2", SAM2_CONFIG_NAME),
              os.path.join(BASE_DIR, "configs", SAM2_CONFIG_NAME)]:
        if os.path.exists(p): SAM2_CONFIG = p; break

# Landmark IDs
LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
NUM_LANDMARKS = 3

LANDMARK_NAMES = {
    0: "NAS (Nasion - front of head)",
    1: "LPA (Left ear)",
    2: "RPA (Right ear)",
}

CONFIG = {
    "frame_skip": 10,
    "flash_duration": 3.0,
    "display_height": 800,
    "duplicate_radius": 40,
    "yolo_conf": 0.25,
    "cap_mask_expansion": 1.10,
    "cap_min_area_frac": 0.03,
    "cap_max_area_frac": 0.70,
    "cap_confirm_alpha": 0.45,
    "cap_autodetect_use_yolo": True,
}

print("=" * 70)


# MODEL LOADING


def load_models():
    """Load YOLO and SAM2 models."""
    
    if not os.path.exists(SAM2_CHECKPOINT):
        print("Downloading SAM2 checkpoint...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        urllib.request.urlretrieve(SAM2_URL, SAM2_CHECKPOINT)
    
    print(f"\nLoading YOLO from {YOLO_WEIGHTS}...")
    if not os.path.exists(YOLO_WEIGHTS):
        print("ERROR: YOLO weights not found!"); sys.exit(1)
    yolo = YOLO(YOLO_WEIGHTS)
    
    print("Loading SAM2 Video Predictor...")
    sam2 = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2 = sam2.float().eval()
    
    print(" Models loaded\n")
    return yolo, sam2


def build_sam2_float32():
    """Build SAM2 for automatic mask generation."""
    model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    return model.float().eval()



# HELPER FUNCTIONS


def resize_for_display(img, target_height):
    """Resize image for display, return (image, scale)."""
    h, w = img.shape[:2]
    scale = target_height / h
    return cv2.resize(img, (int(w * scale), target_height)), scale


def draw_hud(img, idx, total, current_id):
    """Draw heads-up display with status information."""
    
    # Background
    cv2.rectangle(img, (0, 0), (img.shape[1], 170), (0, 0, 0), -1)
    
    # Frame info
    cv2.putText(img, f"Frame: {idx}/{total-1}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Landmark status
    if current_id < NUM_LANDMARKS:
        status = []
        for lid in range(NUM_LANDMARKS):
            name = LANDMARK_NAMES[lid].split(" ")[0]
            if current_id > lid:
                status.append(f"{name}: [Done]")
            elif current_id == lid:
                status.append(f"{name}: [Click]")
            else:
                status.append(f"{name}: [Wait]")
        cv2.putText(img, " | ".join(status), (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Hint for current landmark
        hint = f">>> Click: {LANDMARK_NAMES[current_id]}"
        cv2.putText(img, hint, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        num_electrodes = current_id - NUM_LANDMARKS
        cv2.putText(img, f"Landmarks: [All Done] | Electrodes: {num_electrodes}", 
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Controls
    cv2.putText(img, "NAV: [A] Back (-15) | [S] Fwd (+15) | [Q] Quit", 
                (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, "ACT: [D] YOLO Detect | [R] Refresh Map | [Click] Add | [Space] Done", 
                (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
    cv2.putText(img, "TIP: Use DIFFERENT colored stickers for NAS/LPA/RPA!", 
                (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
    
    return img


def draw_reference_window(img, scale, points, new_detections=None):
    """Draw reference window showing all detected points."""
    
    disp, _ = resize_for_display(img, CONFIG["display_height"])
    ref_img = disp.copy()
    
    # Header
    cv2.rectangle(ref_img, (0, 0), (ref_img.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(ref_img, "DETECTION MAP - Spot missing electrodes", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(ref_img, "GREEN=Landmarks | BLUE=Existing | RED=New YOLO", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw existing points
    for i, (px, py) in enumerate(points):
        dx, dy = int(px * scale), int(py * scale)
        if i < NUM_LANDMARKS:
            # Landmarks - green squares with labels
            cv2.rectangle(ref_img, (dx-6, dy-6), (dx+6, dy+6), (0, 255, 0), -1)
            names = ["N", "L", "R"]
            cv2.putText(ref_img, names[i], (dx+10, dy+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # Electrodes - blue dots
            cv2.circle(ref_img, (dx, dy), 4, (255, 100, 0), -1)
    
    # Draw new YOLO detections
    if new_detections:
        for (x1, y1, x2, y2, eid) in new_detections:
            cx = int(((x1 + x2) / 2) * scale)
            cy = int(((y1 + y2) / 2) * scale)
            cv2.circle(ref_img, (cx, cy), 4, (0, 0, 255), -1)
    
    # Footer
    num_electrodes = max(0, len(points) - NUM_LANDMARKS)
    footer = f"Total: {NUM_LANDMARKS} landmarks, {num_electrodes} electrodes"
    cv2.putText(ref_img, footer, (10, ref_img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return ref_img


def is_duplicate(pt, existing, radius):
    """Check if point is too close to existing electrodes (skip landmarks)."""
    return any(np.linalg.norm(np.array(pt) - np.array(p)) < radius 
               for p in existing[NUM_LANDMARKS:])



# FRAME EXTRACTION


def interactive_crop(video_path, display_height):
    """Let user select crop region."""
    
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx, roi, drawing, start = 0, None, False, None
    
    def mouse(e, x, y, flags, param):
        nonlocal roi, drawing, start
        if e == cv2.EVENT_LBUTTONDOWN:
            drawing, start, roi = True, (x, y), None
        elif e == cv2.EVENT_MOUSEMOVE and drawing:
            roi = (min(start[0], x), min(start[1], y), 
                   abs(x - start[0]), abs(y - start[1]))
        elif e == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    cv2.namedWindow("Crop")
    cv2.setMouseCallback("Crop", mouse)
    
    print("\n--- Crop Selection ---")
    print("Draw a box around the head/cap region")
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        
        disp, scale = resize_for_display(frame, display_height)
        show = disp.copy()
        
        cv2.rectangle(show, (0, 0), (show.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(show, "CROP: Draw box around head/cap", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(show, "A/S: Navigate frames | SPACE: Confirm | Q: Quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        
        if roi:
            cv2.rectangle(show, (roi[0], roi[1]), 
                          (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 255), 2)
        
        cv2.imshow("Crop", show)
        k = cv2.waitKey(30) & 0xFF
        
        if k == ord('a'): idx = max(0, idx - 15)
        elif k == ord('s'): idx = min(total - 1, idx + 15)
        elif k in (13, 32) and roi: break
        elif k == ord('q'): sys.exit(0)
    
    cv2.destroyWindow("Crop")
    cap.release()
    
    return int(roi[0]/scale), int(roi[1]/scale), int(roi[2]/scale), int(roi[3]/scale)


def extract_frames():
    """Extract frames from video."""
    
    if os.path.exists(FRAME_DIR): shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    sx, sy, sw, sh = interactive_crop(VIDEO_PATH, CONFIG["display_height"])
    
    # Validate crop
    if sw < 50 or sh < 50:
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        sx, sy, sw, sh = 0, 0, frame.shape[1], frame.shape[0]
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    skip = CONFIG["frame_skip"]
    frames, idx, count = [], 0, 0
    
    print(f"Extracting frames (skip={skip})...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % skip == 0:
            crop = frame[sy:sy+sh, sx:sx+sw]
            cv2.imwrite(os.path.join(FRAME_DIR, f"{idx:05d}.jpg"), crop)
            frames.append(f"{idx:05d}.jpg")
            idx += 1
        count += 1
    cap.release()
    
    with open(CROP_INFO_FILE, "w") as f:
        json.dump({"x": sx, "y": sy, "w": sw, "h": sh, "skip": skip}, f)
    
    print(f"✓ Extracted {len(frames)} frames")
    return frames


# CAP MASK FUNCTIONS


def expand_mask(mask, expansion=1.10):
    """Expand mask by dilation."""
    if expansion <= 1.0: return mask.astype(bool)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return mask.astype(bool)
    k = max(3, int(max(ys.ptp(), xs.ptp()) * (expansion - 1) * 0.5))
    if k % 2 == 0: k += 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def draw_mask_overlay(disp, mask, alpha=0.45):
    """Draw mask overlay on display image."""
    out = disp.copy()
    m = cv2.resize(mask.astype(np.uint8), (disp.shape[1], disp.shape[0]), 
                   interpolation=cv2.INTER_NEAREST) > 0
    overlay = out.copy()
    overlay[m == 0] = (overlay[m == 0] * 0.35).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def manual_cap_mask(img, sam2, state, fidx, display_h):
    """User clicks center of cap to create mask."""
    disp, scale = resize_for_display(img, display_h)
    clicks = []
    
    def cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x/scale), int(y/scale)))
    
    win = "Click CENTER of cap | q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    
    while not clicks:
        cv2.imshow(win, disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): sys.exit(0)
    cv2.destroyWindow(win)
    
    with torch.inference_mode():
        _, _, logits = sam2.add_new_points_or_box(
            state, frame_idx=fidx, obj_id=999,
            points=[np.array(clicks[0], dtype=np.float32)], labels=[1]
        )
    return (logits[0] > 0).cpu().numpy().squeeze().astype(bool)


def auto_cap_mask(img, yolo=None):
    """Automatically detect cap mask."""
    print("[Auto-Mask] Generating...")
    model = build_sam2_float32()
    amg = SAM2AutomaticMaskGenerator(
        model, points_per_side=32, pred_iou_thresh=0.88,
        stability_score_thresh=0.92, min_mask_region_area=500
    )
    masks = amg.generate(img)
    del amg, model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    
    H, W = img.shape[:2]
    amin = H * W * CONFIG["cap_min_area_frac"]
    amax = H * W * CONFIG["cap_max_area_frac"]
    
    centers = []
    if CONFIG["cap_autodetect_use_yolo"] and yolo:
        res = yolo.predict(img, conf=0.2, verbose=False)
        if res and res[0].boxes is not None:
            centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) 
                       for b in res[0].boxes.xyxy.cpu().numpy()]
    
    best, score = None, -1e9
    for m in masks:
        area = float(m["segmentation"].sum())
        if not (amin <= area <= amax): continue
        s = area + sum(1000 for cx, cy in centers 
                       if 0 <= int(cy) < H and 0 <= int(cx) < W 
                       and m["segmentation"][int(cy), int(cx)])
        if s > score: best, score = m["segmentation"].astype(bool), s
    
    if best is not None:
        return best
    return max(masks, key=lambda x: x["area"])["segmentation"].astype(bool) if masks else np.ones((H, W), bool)


def create_cap_mask(img, sam2, state, yolo):
    """Create cap mask with user confirmation."""
    mode = "manual"
    
    while True:
        if mode == "auto":
            mask = auto_cap_mask(img, yolo)
        else:
            mask = manual_cap_mask(img, sam2, state, 0, CONFIG["display_height"])
        
        mask_exp = expand_mask(mask, CONFIG["cap_mask_expansion"])
        disp, _ = resize_for_display(img, CONFIG["display_height"])
        preview = draw_mask_overlay(disp, mask_exp, CONFIG["cap_confirm_alpha"])
        
        win = "Confirm Cap Mask"
        cv2.namedWindow(win)
        
        while True:
            show = preview.copy()
            cv2.rectangle(show, (0, 0), (show.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(show, f"Mode={mode} | y=OK | r=redo | m=manual | a=auto | q=quit", 
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(show, "Tip: Press 'm' and click center of cap", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
            cv2.imshow(win, show)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('y'):
                with torch.inference_mode():
                    sam2.add_new_mask(state, frame_idx=0, obj_id=999, 
                                      mask=mask_exp.astype(np.uint8))
                cv2.destroyWindow(win)
                return mask_exp
            if k == ord('r'): break
            if k == ord('m'): mode = "manual"; break
            if k == ord('a'): mode = "auto"; break
            if k == ord('q'): sys.exit(0)
        cv2.destroyWindow(win)


def precompute_cap_masks(sam2, state, frames):
    """Pre-track cap mask through all frames."""
    print("Pre-tracking cap mask...")
    masks = {}
    
    with torch.inference_mode():
        for fidx, ids, logits in tqdm(sam2.propagate_in_video(state), total=len(frames)):
            if 999 in ids:
                m = (logits[list(ids).index(999)] > 0).cpu().numpy().squeeze()
                masks[fidx] = expand_mask(m, CONFIG["cap_mask_expansion"])
            else:
                masks[fidx] = None
    return masks



# MAIN PIPELINE


def main():
    print("\n" + "=" * 70)
    print("EEG ELECTRODE REGISTRATION - SCRIPT 1")
    print("Detection & Tracking")
    print("=" * 70 + "\n")

    # 1. Load models
    yolo, sam2 = load_models()

    # 2. Extract frames
    frames = extract_frames()
    if not frames: return

    # 3. Initialize SAM2 state
    print("\nInitializing SAM2 state...")
    with torch.inference_mode():
        state = sam2.init_state(video_path=FRAME_DIR)

    # 4. Create cap mask
    first_img = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
    cap_mask = create_cap_mask(first_img, sam2, state, yolo)

    # 5. Pre-compute cap masks
    print("\nPre-computing cap masks...")
    with torch.inference_mode():
        cap_state = sam2.init_state(video_path=FRAME_DIR)
        sam2.add_new_mask(cap_state, frame_idx=0, obj_id=999, mask=cap_mask.astype(np.uint8))
    cap_cache = precompute_cap_masks(sam2, cap_state, frames)
    del cap_state; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # 6. Interactive labeling
    _, SCALE = resize_for_display(first_img, CONFIG["display_height"])
    
    points = []
    current_id = 0
    idx_box = [0]
    flashes = []

    def click(e, x, y, f, p):
        nonlocal current_id
        if e != cv2.EVENT_LBUTTONDOWN: return
        
        cidx = idx_box[0]
        rx, ry = int(x/SCALE), int(y/SCALE)

        # For electrodes, check cap mask and duplicates
        cap = cap_cache.get(cidx)
        if current_id >= NUM_LANDMARKS:
            if cap is None or not (0 <= ry < cap.shape[0] and 0 <= rx < cap.shape[1] and cap[ry, rx]):
                print(">>> Outside cap mask - rejected")
                return
            if is_duplicate((rx, ry), points, CONFIG["duplicate_radius"]):
                print(">>> Too close to existing electrode - rejected")
                return

        with torch.inference_mode():
            sam2.add_new_points_or_box(
                state, frame_idx=cidx, obj_id=current_id,
                points=[np.array((rx, ry), dtype=np.float32)], labels=[1]
            )
        
        points.append((rx, ry))
        color = (0, 255, 0) if current_id < NUM_LANDMARKS else (0, 0, 255)
        flashes.append((rx, ry, color, time.time()))
        
        if current_id < NUM_LANDMARKS:
            print(f"Added landmark {LANDMARK_NAMES[current_id]} at ({rx}, {ry})")
        else:
            print(f"Added electrode E{current_id - NUM_LANDMARKS} at ({rx}, {ry})")
        current_id += 1

    cv2.namedWindow("Pipeline")
    cv2.setMouseCallback("Pipeline", click)

    print("\n--- Interactive Labeling ---")
    print(f"1. Click {NUM_LANDMARKS} landmarks: NAS (red), LPA (blue), RPA (green)")
    print("2. Press 'D' to auto-detect electrodes with YOLO")
    print("3. Click manually to add missed electrodes")
    print("4. Press 'R' to refresh the reference map")
    print("5. Press SPACE when done")
    
    while True:
        cidx = idx_box[0]
        img = cv2.imread(os.path.join(FRAME_DIR, frames[cidx]))
        if img is None: break
        
        disp, _ = resize_for_display(img, CONFIG["display_height"])

        # Overlay cap mask
        cap = cap_cache.get(cidx)
        if cap is not None:
            overlay = disp.copy()
            rm = cv2.resize(cap.astype(np.uint8), (disp.shape[1], disp.shape[0]))
            overlay[rm == 0] = overlay[rm == 0] // 2
            disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)

        # Draw flashes
        now = time.time()
        flashes[:] = [f for f in flashes if now - f[3] < CONFIG["flash_duration"]]
        for fx, fy, col, _ in flashes:
            cv2.circle(disp, (int(fx*SCALE), int(fy*SCALE)), 7, col, -1)

        disp = draw_hud(disp, cidx, len(frames), current_id)
        cv2.imshow("Pipeline", disp)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('s'):
            idx_box[0] = min(cidx + 15, len(frames) - 1)
        elif k == ord('a'):
            idx_box[0] = max(cidx - 15, 0)
        elif k == ord('q'):
            sys.exit(0)
            
        elif k == ord('d'):
            if current_id < NUM_LANDMARKS:
                print(f">>> Click all {NUM_LANDMARKS} landmarks first!")
                continue
            
            print(f"\nRunning YOLO detection on frame {cidx}...")
            res = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
            
            new_detections = []
            found = 0
            
            if res and res[0].boxes is not None:
                cap = cap_cache.get(cidx)
                for box in res[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    if cap is not None:
                        if not (0 <= int(cy) < cap.shape[0] and 
                                0 <= int(cx) < cap.shape[1] and cap[int(cy), int(cx)]):
                            continue
                    
                    if is_duplicate((cx, cy), points, CONFIG["duplicate_radius"]):
                        continue
                    
                    with torch.inference_mode():
                        sam2.add_new_points_or_box(
                            state, frame_idx=cidx, obj_id=current_id, box=box
                        )
                    
                    points.append((cx, cy))
                    new_detections.append((x1, y1, x2, y2, current_id))
                    flashes.append((cx, cy, (0, 0, 255), time.time()))
                    current_id += 1
                    found += 1
            
            print(f"Added {found} new electrodes")
            ref_img = draw_reference_window(img, SCALE, points, new_detections)
            cv2.imshow("Reference Map", ref_img)
            
        elif k == ord('r'):
            if current_id >= NUM_LANDMARKS:
                ref_img = draw_reference_window(img, SCALE, points, None)
                cv2.imshow("Reference Map", ref_img)
                
        elif k == 32:  # Space
            if current_id < NUM_LANDMARKS:
                print(f">>> Must click all {NUM_LANDMARKS} landmarks first")
                continue
            try: cv2.destroyWindow("Reference Map")
            except: pass
            break

    cv2.destroyAllWindows()

    # 7. SAM2 Tracking
    print("\n--- SAM2 Tracking ---")
    print(f"Tracking {current_id} objects across {len(frames)} frames...")
    
    tracking = {}
    with torch.inference_mode():
        for fidx, ids, logits in tqdm(sam2.propagate_in_video(state), total=len(frames)):
            fd = {}
            for i, oid in enumerate(ids):
                if oid == 999: continue
                m = (logits[i] > 0).cpu().numpy().squeeze()
                ys, xs = np.where(m)
                if len(xs) > 0:
                    fd[int(oid)] = (float(np.mean(xs)), float(np.mean(ys)))
            tracking[int(fidx)] = fd

    # 8. Save results
    print("\nSaving results...")
    with open(TRACKING_FILE, "wb") as f:
        pickle.dump(tracking, f)
    print(f"✓ Saved: {TRACKING_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("SCRIPT 1 COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Frames: {FRAME_DIR}/ ({len(frames)} frames)")
    print(f"  - Tracking: {TRACKING_FILE}")
    print(f"  - Crop info: {CROP_INFO_FILE}")
    print(f"\nStatistics:")
    print(f"  - Landmarks: {NUM_LANDMARKS}")
    print(f"  - Electrodes: {current_id - NUM_LANDMARKS}")
    print(f"  - Frames tracked: {len(tracking)}")
    print(f"\nNext: Run Script 2 (VGGT 3D reconstruction)")


if __name__ == "__main__":
    main()