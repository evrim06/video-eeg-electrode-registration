"""
SCRIPT 1 

APPROACH:
    - VGGT helps SAM2 track (recover lost electrodes, visibility check)
    - Multiple prompts + negative prompts for robust tracking
    - Output: 2D tracking positions + masks 

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
import glob
import urllib.request
from tqdm import tqdm
from contextlib import nullcontext
from ultralytics import YOLO
import colorsys



# SAM2 DTYPE PATCH

print("SCRIPT 1: VGGT-ASSISTED TRACKING WITH SAM2 + YOLO")



torch.set_default_dtype(torch.float32)

import sam2
from sam2.modeling.memory_attention import MemoryAttentionLayer, MemoryAttention
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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
print(f" Device: {DEVICE.upper()}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

VIDEO_DIR = os.path.join(BASE_DIR, "data", "Video_Recordings")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VGGT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "vggt_output")
TEMP_VGGT_DIR = os.path.join(BASE_DIR, "data", "vggt_ready_frames")
VGGT_REPO_PATH = os.path.join(BASE_DIR, "vggt")

# Output files (same as original Script1 - compatible with Script3)
TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
CROP_INFO_FILE = os.path.join(RESULTS_DIR, "crop_info.json")
MASKS_CACHE_FILE = os.path.join(RESULTS_DIR, "masks_cache.pkl")
# NO electrodes_3d.json here! Script3 computes that.

YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train4", "weights", "best.pt")
SAM2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam2_hiera_small.pt")
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
SAM2_CONFIG_NAME = "sam2_hiera_s.yaml"

SAM2_CONFIG = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2", SAM2_CONFIG_NAME)
if not os.path.exists(SAM2_CONFIG):
    for p in [os.path.join(BASE_DIR, "configs", "sam2", SAM2_CONFIG_NAME),
              os.path.join(BASE_DIR, "configs", SAM2_CONFIG_NAME)]:
        if os.path.exists(p): SAM2_CONFIG = p; break

# IDs (same as original - compatible with Script3)
CAP_MASK_ID = 999
LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
NUM_LANDMARKS = 3
ELECTRODE_START_ID = 100

LANDMARK_NAMES = {0: "NAS (Nasion)", 1: "LPA (Left ear)", 2: "RPA (Right ear)"}
LANDMARK_SHORT = {0: "NAS", 1: "LPA", 2: "RPA"}

VGGT_SIZE = 518
MAX_VGGT_FRAMES = 35

CONFIG = {
    "frame_skip": 10,
    "display_height": 600,  # Fits laptop screens
    "duplicate_radius": 40,
    "yolo_conf": 0.25,
    "mask_alpha": 0.5,
    "playback_fps": 15,
    "cap_mask_expansion": 1.10,
    "min_cap_overlap": 0.10,
    "visibility_depth_threshold": 0.3,
    "negative_prompt_distance": 40,
}

print(f" VGGT keyframes: {MAX_VGGT_FRAMES}")




# COLORS


def get_color(obj_id):
    colors = {
        LANDMARK_NAS: (0, 0, 255), LANDMARK_LPA: (255, 0, 0), 
        LANDMARK_RPA: (0, 255, 0), CAP_MASK_ID: (0, 255, 255)
    }
    if obj_id in colors:
        return colors[obj_id]
    i = obj_id - ELECTRODE_START_ID if obj_id >= ELECTRODE_START_ID else obj_id
    hue = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return (int(b * 255), int(g * 255), int(r * 255))



# VIDEO SELECTION


def select_video():
    extensions = ['*.mp4', '*.MP4', '*.MOV', '*.mov']
    all_videos = []
    for ext in extensions:
        all_videos.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    
    seen = set()
    videos = []
    for v in sorted(all_videos):
        if v.lower() not in seen:
            seen.add(v.lower())
            videos.append(v)
    
    if not videos:
        print(f"ERROR: No videos in {VIDEO_DIR}")
        sys.exit(1)
    
    print("\n--- Select Video ---")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {os.path.basename(v)}")
    
    choice = input(f"\nEnter (1-{len(videos)}): ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(videos):
            return videos[idx]
    except:
        pass
    return videos[0]


# HELPERS


def resize_for_display(img, h):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h)), scale


def get_mask_centroid(mask):
    if mask is None:
        return None
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return None
    ys, xs = np.where(mask_bool)
    return float(np.mean(xs)), float(np.mean(ys))


def mask_inside_cap_ratio(mask, cap_mask):
    if mask is None or cap_mask is None:
        return 0.0
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return 0.0
    cap_bool = cap_mask > 0
    if mask_bool.shape != cap_bool.shape:
        cap_bool = cv2.resize(cap_mask.astype(np.uint8), (mask.shape[1], mask.shape[0])) > 0
    return (mask_bool & cap_bool).sum() / mask_bool.sum()


def draw_mask_overlay(img, mask, color, alpha=0.5):
    if mask is None:
        return img
    mask_uint8 = (mask > 0).astype(np.uint8)
    if mask_uint8.sum() == 0:
        return img
    overlay = img.copy()
    if mask_uint8.shape[:2] != img.shape[:2]:
        mask_uint8 = cv2.resize(mask_uint8, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = mask_uint8 > 0
    overlay[m] = (overlay[m] * (1-alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def draw_all_masks(img, masks_dict):
    result = img.copy()
    for obj_id, mask in masks_dict.items():
        if obj_id == CAP_MASK_ID or mask is None:
            continue
        color = get_color(obj_id)
        result = draw_mask_overlay(result, mask, color, CONFIG["mask_alpha"])
        centroid = get_mask_centroid(mask)
        if centroid:
            cx, cy = int(centroid[0]), int(centroid[1])
            mask_bool = mask > 0
            if mask_bool.shape[:2] != result.shape[:2]:
                sy, sx = result.shape[0] / mask_bool.shape[0], result.shape[1] / mask_bool.shape[1]
                cx, cy = int(cx * sx), int(cy * sy)
            label = LANDMARK_SHORT.get(obj_id, f"E{obj_id - ELECTRODE_START_ID}")
            cv2.rectangle(result, (cx-15, cy-10), (cx+15, cy+5), (0,0,0), -1)
            cv2.putText(result, label, (cx-12, cy+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return result


def expand_mask(mask, exp=1.10):
    mask_bool = mask > 0
    if exp <= 1.0:
        return mask_bool
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return mask_bool
    k = max(3, int(max(ys.ptp(), xs.ptp()) * (exp - 1) * 0.5))
    if k % 2 == 0:
        k += 1
    return cv2.dilate(mask_bool.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


# COORDINATE TRANSFORMS 

def script1_to_vggt(u, v, crop_w, crop_h):
    """Convert Script1 pixel coords to VGGT coords."""
    scale = VGGT_SIZE / max(crop_w, crop_h)
    pad_w = (VGGT_SIZE - int(crop_w * scale)) // 2
    pad_h = (VGGT_SIZE - int(crop_h * scale)) // 2
    return u * scale + pad_w, v * scale + pad_h


def vggt_to_script1(u, v, crop_w, crop_h):
    """Convert VGGT coords to Script1 pixel coords."""
    scale = VGGT_SIZE / max(crop_w, crop_h)
    pad_w = (VGGT_SIZE - int(crop_w * scale)) // 2
    pad_h = (VGGT_SIZE - int(crop_h * scale)) // 2
    return (u - pad_w) / scale, (v - pad_h) / scale


def unproject_to_3d(u, v, depth, K, E):
    """Convert 2D pixel to 3D point (for tracking guidance only)."""
    H, W = depth.shape
    if not (0 <= u < W-1 and 0 <= v < H-1):
        return None
    z = depth[int(v), int(u)]
    if z <= 0:
        return None
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    P = np.array([x, y, z, 1.0])
    return (np.linalg.inv(E) @ P)[:3]


def project_to_2d(point_3d, K, E):
    """Project 3D point to 2D pixel (for tracking guidance)."""
    P = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    P_cam = E @ P
    if P_cam[2] <= 0:
        return None
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = fx * P_cam[0] / P_cam[2] + cx
    v = fy * P_cam[1] / P_cam[2] + cy
    if not (0 <= u < VGGT_SIZE and 0 <= v < VGGT_SIZE):
        return None
    return (u, v)


def check_visibility(point_3d, depth, K, E, threshold=0.3):
    """Check if 3D point is visible (for tracking guidance)."""
    uv = project_to_2d(point_3d, K, E)
    if uv is None:
        return False, None
    
    u, v = uv
    observed_depth = depth[int(v), int(u)]
    
    P = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    expected_depth = (E @ P)[2]
    
    if expected_depth <= 0 or observed_depth <= 0:
        return False, None
    
    depth_diff = abs(observed_depth - expected_depth) / expected_depth
    return depth_diff < threshold, uv


# NEGATIVE PROMPTS (Safe placement)


def get_negative_points_safe(cx, cy, cap_mask, img_h, img_w, all_electrode_positions, 
                             n_points=4, min_distance=15, max_distance=40, safe_margin=20):
    """Generate safe negative prompt points that don't overlap with other electrodes."""
    negative_points = []
    cap_bool = cap_mask > 0 if cap_mask is not None else None
    
    def is_safe_from_electrodes(px, py):
        for ex, ey in all_electrode_positions:
            dist = np.sqrt((px - ex)**2 + (py - ey)**2)
            if dist < safe_margin:
                return False
        return True
    
    def is_outside_cap(px, py):
        if cap_bool is None:
            return False
        if 0 <= int(py) < cap_bool.shape[0] and 0 <= int(px) < cap_bool.shape[1]:
            return not cap_bool[int(py), int(px)]
        return True
    
    # PRIORITY 1: Points OUTSIDE cap (safest)
    for distance in range(min_distance, max_distance + 1, 5):
        offsets = [
            (0, -distance), (0, distance),
            (-distance, 0), (distance, 0),
            (-distance, -distance), (distance, -distance),
            (-distance, distance), (distance, distance),
        ]
        
        for dx, dy in offsets:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < img_w and 0 <= ny < img_h):
                continue
            if is_outside_cap(nx, ny):
                negative_points.append((nx, ny))
                if len(negative_points) >= n_points:
                    return negative_points
    
    # PRIORITY 2: Points inside cap but safe from electrodes
    for distance in range(min_distance, max_distance + 1, 5):
        offsets = [(0, -distance), (0, distance), (-distance, 0), (distance, 0)]
        for dx, dy in offsets:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < img_w and 0 <= ny < img_h):
                continue
            if is_safe_from_electrodes(nx, ny):
                negative_points.append((nx, ny))
                if len(negative_points) >= n_points:
                    return negative_points
    
    return negative_points


def get_negative_points(cx, cy, cap_mask, img_h, img_w, n_points=4, distance=30):
    """Simple version for single electrode."""
    negative_points = []
    cap_bool = cap_mask > 0 if cap_mask is not None else None
    
    for dist in [distance, distance + 10, distance + 20]:
        offsets = [(0, -dist), (0, dist), (-dist, 0), (dist, 0)]
        for dx, dy in offsets:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < img_w and 0 <= ny < img_h):
                continue
            if cap_bool is not None:
                if 0 <= int(ny) < cap_bool.shape[0] and 0 <= int(nx) < cap_bool.shape[1]:
                    if not cap_bool[int(ny), int(nx)]:
                        negative_points.append((nx, ny))
                        if len(negative_points) >= n_points:
                            return negative_points
    return negative_points



# MODEL LOADING


def load_models():
    print(f"\nLoading models...")
    
    if not os.path.exists(YOLO_WEIGHTS):
        print("ERROR: YOLO not found"); sys.exit(1)
    yolo = YOLO(YOLO_WEIGHTS)
    
    if not os.path.exists(SAM2_CHECKPOINT):
        print("Downloading SAM2...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        urllib.request.urlretrieve(SAM2_URL, SAM2_CHECKPOINT)
    
    sam2_video = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_video = sam2_video.float().eval()
    
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_image = SAM2ImagePredictor(sam2_model.float().eval())
    
    print("[OK] Models loaded")
    return yolo, sam2_video, sam2_image



# FRAME EXTRACTION


def extract_frames(video_path):
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx, roi, drawing, start = 0, None, False, None
    
    def mouse(e, x, y, f, p):
        nonlocal roi, drawing, start
        if e == cv2.EVENT_LBUTTONDOWN:
            drawing, start, roi = True, (x,y), None
        elif e == cv2.EVENT_MOUSEMOVE and drawing:
            roi = (min(start[0],x), min(start[1],y), abs(x-start[0]), abs(y-start[1]))
        elif e == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    cv2.namedWindow("Crop")
    cv2.setMouseCallback("Crop", mouse)
    print("\n--- Draw box around HEAD ---")
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        disp, scale = resize_for_display(frame, CONFIG["display_height"])
        show = disp.copy()
        cv2.putText(show, f"Frame {idx}/{total-1} | A/S:Nav | SPACE:Confirm", (10,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if roi:
            cv2.rectangle(show, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0,255,255), 2)
        cv2.imshow("Crop", show)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('a'): idx = max(0, idx-15)
        elif k == ord('s'): idx = min(total-1, idx+15)
        elif k in (13,32) and roi: break
        elif k == ord('q'): sys.exit(0)
    
    cv2.destroyWindow("Crop")
    cap.release()
    
    sx, sy, sw, sh = int(roi[0]/scale), int(roi[1]/scale), int(roi[2]/scale), int(roi[3]/scale)
    
    cap = cv2.VideoCapture(video_path)
    skip = CONFIG["frame_skip"]
    frames, fidx, count = [], 0, 0
    print(f"\nExtracting frames (skip={skip})...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % skip == 0:
            crop = frame[sy:sy+sh, sx:sx+sw]
            cv2.imwrite(os.path.join(FRAME_DIR, f"{fidx:05d}.jpg"), crop)
            frames.append(f"{fidx:05d}.jpg")
            fidx += 1
        count += 1
    cap.release()
    
    crop_info = {"x": sx, "y": sy, "w": sw, "h": sh, "skip": skip}
    with open(CROP_INFO_FILE, "w") as f:
        json.dump(crop_info, f)
    
    print(f"Extracted {len(frames)} frames")
    return frames, crop_info


# VGGT (saves reconstruction.npz for Script3)


def run_vggt(frames, crop_info):
    print(f"\n--- Running VGGT ({MAX_VGGT_FRAMES} keyframes) ---")
    
    sys.path.insert(0, VGGT_REPO_PATH)
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
    os.makedirs(VGGT_OUTPUT_DIR, exist_ok=True)
    
    all_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
    n = min(len(all_paths), MAX_VGGT_FRAMES)
    indices = np.linspace(0, len(all_paths)-1, n, dtype=int)
    selected = [all_paths[i] for i in indices]
    s1_indices = [int(os.path.basename(p).split('.')[0]) for p in selected]
    
    avg_gap = len(all_paths) / n
    print(f"  Keyframes: {len(s1_indices)} (every ~{avg_gap:.1f} frames)")
    
    if os.path.exists(TEMP_VGGT_DIR): shutil.rmtree(TEMP_VGGT_DIR)
    os.makedirs(TEMP_VGGT_DIR)
    
    processed = []
    for i, path in enumerate(selected):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        scale = VGGT_SIZE / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        pad_w = (VGGT_SIZE - img.shape[1]) // 2
        pad_h = (VGGT_SIZE - img.shape[0]) // 2
        img = cv2.copyMakeBorder(img, pad_h, VGGT_SIZE-img.shape[0]-pad_h,
                                 pad_w, VGGT_SIZE-img.shape[1]-pad_w,
                                 cv2.BORDER_CONSTANT, value=(0,0,0))
        out = os.path.join(TEMP_VGGT_DIR, f"{i:05d}.jpg")
        cv2.imwrite(out, img)
        processed.append(out)
    
    checkpoint = os.path.join(CHECKPOINT_DIR, "vggt_model.pt")
    model = VGGT()
    if not os.path.exists(checkpoint):
        print("  Downloading VGGT weights...")
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=DEVICE, progress=True)
        torch.save(state, checkpoint)
    
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    images = load_and_preprocess_images(processed).to(DEVICE)
    ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if DEVICE == "cuda" else nullcontext()
    
    print("  Running VGGT inference...")
    with torch.no_grad(), ctx:
        preds = model(images)
    
    E, K = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    depth = preds["depth"].float().cpu().numpy().squeeze()
    E = E.float().cpu().numpy().squeeze()
    K = K.float().cpu().numpy().squeeze()
    
    if E.shape[-2:] == (3,4):
        E4 = np.zeros((len(s1_indices), 4, 4))
        for i in range(len(s1_indices)):
            E4[i] = np.eye(4)
            E4[i,:3,:] = E[i]
        E = E4
    
    # Save for Script3 (same format as original Script2)
    np.savez_compressed(os.path.join(VGGT_OUTPUT_DIR, "reconstruction.npz"),
                       depth=depth, 
                       extrinsics=E, 
                       intrinsics=K,
                       frame_mapping_keys=np.arange(len(s1_indices)),
                       frame_mapping_values=np.array(s1_indices))
    
    del model, images, preds
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    
    print(f"[OK] VGGT complete - reconstruction.npz saved for Script3")
    return {"depth": depth, "extrinsics": E, "intrinsics": K,
            "s1_indices": s1_indices, "s1_to_vggt": {s:i for i,s in enumerate(s1_indices)}}



# CAP MASK


def create_cap_mask(frames, sam2_video, sam2_image):
    print("\n--- Cap Mask ---")
    print("Click CENTER of cap")
    
    img = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
    disp, scale = resize_for_display(img, CONFIG["display_height"])
    
    clicks = []
    cap = None
    
    def mouse(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x/scale), int(y/scale)))
    
    cv2.namedWindow("Cap")
    cv2.setMouseCallback("Cap", mouse)
    sam2_image.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    while True:
        show = disp.copy()
        if cap is not None:
            show = draw_mask_overlay(show, cap, (0,255,255), 0.4)
            cv2.putText(show, "Y:Accept R:Redo", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(show, "Click cap center", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Cap", show)
        k = cv2.waitKey(1) & 0xFF
        
        if clicks and cap is None:
            with torch.inference_mode():
                masks, scores, _ = sam2_image.predict(
                    point_coords=np.array([clicks[-1]]), point_labels=np.array([1]), multimask_output=True)
            cap = expand_mask(masks[np.argmax(scores)], CONFIG["cap_mask_expansion"])
            clicks.clear()
        
        if k == ord('y') and cap is not None: break
        elif k == ord('r'): cap = None
        elif k == ord('q'): sys.exit(0)
    
    cv2.destroyWindow("Cap")
    
    print("  Tracking cap mask...")
    with torch.inference_mode():
        state = sam2_video.init_state(video_path=FRAME_DIR)
        sam2_video.add_new_mask(state, frame_idx=0, obj_id=CAP_MASK_ID, mask=cap.astype(np.uint8))
        
        cap_masks = {}
        for fidx, ids, logits in tqdm(sam2_video.propagate_in_video(state), total=len(frames), desc="  Cap"):
            if CAP_MASK_ID in ids:
                m = (logits[list(ids).index(CAP_MASK_ID)] > 0).cpu().numpy().squeeze()
                cap_masks[fidx] = expand_mask(m, CONFIG["cap_mask_expansion"])
    
    print(f"Cap tracked")
    return cap_masks



# VGGT-ASSISTED TRACKING


def vggt_assisted_tracking(frames, cap_masks, yolo, sam2_video, sam2_image,
                            vggt_data, crop_info):
    """
    Track electrodes with SAM2, use VGGT to recover lost ones.
    Output: 2D positions only (Script3 does 3D).
    """
    
    print("VGGT-ASSISTED TRACKING")
    print("SAM2 tracks | VGGT recovers lost electrodes")

    
    vggt_frames = sorted(vggt_data["s1_to_vggt"].keys())
    
    # Detection in middle keyframe
    mid_idx = len(vggt_frames) // 2
    det_frame = vggt_frames[mid_idx]
    
    print(f"\n  Detection frame: {det_frame}")
    
    img = cv2.imread(os.path.join(FRAME_DIR, frames[det_frame]))
    img_h, img_w = img.shape[:2]
    cap = cap_masks.get(det_frame)
    cap_bool = cap > 0 if cap is not None else None
    
    res = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
    
    detections = []
    if res and res[0].boxes is not None:
        for box in res[0].boxes.xyxy.cpu().numpy():
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            if cap_bool is not None:
                if not (0 <= int(cy) < cap_bool.shape[0] and 
                        0 <= int(cx) < cap_bool.shape[1] and cap_bool[int(cy), int(cx)]):
                    continue
            detections.append((cx, cy, box))
    
    print(f"  YOLO detected: {len(detections)} electrodes")
    
    # Get 3D positions for tracking guidance (NOT for final output)
    vi = vggt_data["s1_to_vggt"][det_frame]
    depth = vggt_data["depth"][vi]
    K = vggt_data["intrinsics"][vi]
    E = vggt_data["extrinsics"][vi]
    
    electrodes_3d_guidance = {}  # For tracking help only
    electrode_ids = []
    
    for i, (cx, cy, box) in enumerate(detections):
        obj_id = ELECTRODE_START_ID + i
        electrode_ids.append(obj_id)
        
        uv = script1_to_vggt(cx, cy, crop_info["w"], crop_info["h"])
        p3d = unproject_to_3d(uv[0], uv[1], depth, K, E)
        if p3d is not None:
            electrodes_3d_guidance[obj_id] = p3d
    
    print(f"  3D guidance positions: {len(electrodes_3d_guidance)}")
    
    # Initialize tracking
    all_masks = {f: {} for f in range(len(frames))}
    tracking = {f: {} for f in range(len(frames))}
    
    total_recoveries = 0
    all_electrode_positions = [(cx, cy) for cx, cy, box in detections]
    
    with torch.inference_mode():
        state = sam2_video.init_state(video_path=FRAME_DIR)
        
        # Add cap
        if 0 in cap_masks:
            sam2_video.add_new_mask(state, frame_idx=0, obj_id=CAP_MASK_ID,
                                   mask=(cap_masks[0] > 0).astype(np.uint8))
        
        # Add electrodes with robust prompts
        for i, (cx, cy, box) in enumerate(detections):
            obj_id = ELECTRODE_START_ID + i
            
            neg_points = get_negative_points_safe(
                cx, cy, cap, img_h, img_w, all_electrode_positions,
                n_points=4, min_distance=15, max_distance=CONFIG["negative_prompt_distance"],
                safe_margin=CONFIG["duplicate_radius"]
            )
            
            if neg_points:
                all_points = [(cx, cy)] + neg_points
                all_labels = [1] + [0] * len(neg_points)
                sam2_video.add_new_points_or_box(
                    state, frame_idx=det_frame, obj_id=obj_id, box=box,
                    points=np.array(all_points, dtype=np.float32),
                    labels=np.array(all_labels, dtype=np.int32)
                )
            else:
                sam2_video.add_new_points_or_box(
                    state, frame_idx=det_frame, obj_id=obj_id, box=box,
                    points=np.array([(cx, cy)], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32)
                )
        
        print(f"  Tracking with VGGT recovery at {len(vggt_frames)} keyframes...")
        
        for fidx, ids, logits in tqdm(sam2_video.propagate_in_video(state), 
                                      total=len(frames), desc="  Tracking"):
            cap = cap_masks.get(fidx)
            cap_bool = cap > 0 if cap is not None else None
            
            frame_masks = {}
            frame_tracking = {}
            
            for i, oid in enumerate(ids):
                if oid == CAP_MASK_ID:
                    continue
                
                mask = (logits[i] > 0).cpu().numpy().squeeze()
                
                if cap_bool is not None:
                    overlap = mask_inside_cap_ratio(mask, cap_bool)
                    if overlap < CONFIG["min_cap_overlap"]:
                        continue
                
                centroid = get_mask_centroid(mask)
                if centroid:
                    frame_masks[int(oid)] = mask
                    frame_tracking[int(oid)] = centroid
            
            all_masks[fidx] = frame_masks
            tracking[fidx] = frame_tracking
            currently_tracked = set(frame_tracking.keys())
            
            # VGGT recovery at keyframes
            if fidx in vggt_data["s1_to_vggt"]:
                vi = vggt_data["s1_to_vggt"][fidx]
                kf_depth = vggt_data["depth"][vi]
                kf_K = vggt_data["intrinsics"][vi]
                kf_E = vggt_data["extrinsics"][vi]
                
                for obj_id in electrode_ids:
                    if obj_id in currently_tracked:
                        continue
                    if obj_id not in electrodes_3d_guidance:
                        continue
                    
                    is_visible, uv_vggt = check_visibility(
                        electrodes_3d_guidance[obj_id], kf_depth, kf_K, kf_E,
                        CONFIG["visibility_depth_threshold"]
                    )
                    
                    if not is_visible or uv_vggt is None:
                        continue
                    
                    pred_x, pred_y = vggt_to_script1(
                        uv_vggt[0], uv_vggt[1], crop_info["w"], crop_info["h"]
                    )
                    
                    if cap_bool is not None:
                        if not (0 <= int(pred_y) < cap_bool.shape[0] and 
                                0 <= int(pred_x) < cap_bool.shape[1] and 
                                cap_bool[int(pred_y), int(pred_x)]):
                            continue
                    
                    current_positions = [(x, y) for x, y in frame_tracking.values()]
                    neg_points = get_negative_points_safe(
                        pred_x, pred_y, cap, img_h, img_w, current_positions,
                        n_points=4, min_distance=15, max_distance=CONFIG["negative_prompt_distance"],
                        safe_margin=CONFIG["duplicate_radius"]
                    )
                    
                    try:
                        if neg_points:
                            all_points = [(pred_x, pred_y)] + neg_points
                            all_labels = [1] + [0] * len(neg_points)
                            sam2_video.add_new_points_or_box(
                                state, frame_idx=fidx, obj_id=obj_id,
                                points=np.array(all_points, dtype=np.float32),
                                labels=np.array(all_labels, dtype=np.int32)
                            )
                        else:
                            sam2_video.add_new_points_or_box(
                                state, frame_idx=fidx, obj_id=obj_id,
                                points=np.array([(pred_x, pred_y)], dtype=np.float32),
                                labels=np.array([1], dtype=np.int32)
                            )
                        total_recoveries += 1
                    except:
                        pass
    
    total_tracked = sum(len(m) for m in all_masks.values())
    avg = total_tracked / len(frames) if frames else 0
    
    print(f"\nTracking complete")
    print(f"  Average electrodes/frame: {avg:.1f}")
    print(f"  VGGT recoveries: {total_recoveries}")
    
    return all_masks, tracking, electrode_ids


# INTERACTIVE LANDMARK ADDITION


def interactive_add(frames, all_masks, tracking, cap_masks, sam2_video, sam2_image, 
                    vggt_data, crop_info):
    """User adds landmarks + any missed electrodes."""
    

    print("ADD LANDMARKS")
    print("\nSteps:")
    print("  1. Front view -> click NAS (red)")
    print("  2. Left view -> click LPA (blue)")
    print("  3. Right view -> click RPA (green)")
    print("  4. (Optional) Click missed electrodes")
    print("  5. Press T to track new additions")
    print("  6. SPACE when done")
    
    idx = 0
    playing = False
    last_t = time.time()
    
    lm_done = [False, False, False]
    curr_lm = 0
    next_id = ELECTRODE_START_ID + 100  # Leave room for YOLO detections
    
    new_adds = []
    
    img0 = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
    img_h, img_w = img0.shape[:2]
    _, SCALE = resize_for_display(img0, CONFIG["display_height"])
    
    vggt_frames_set = set(vggt_data["s1_to_vggt"].keys())
    
    def add_obj(rx, ry, fidx, is_lm):
        nonlocal curr_lm, next_id
        
        if is_lm and curr_lm < NUM_LANDMARKS:
            oid = curr_lm
            lm_done[curr_lm] = True
            print(f"  [OK] {LANDMARK_NAMES[curr_lm]} at frame {fidx}")
            curr_lm += 1
        else:
            oid = next_id
            print(f"  [OK] E{oid - ELECTRODE_START_ID} at frame {fidx}")
            next_id += 1
        
        img = cv2.imread(os.path.join(FRAME_DIR, frames[fidx]))
        sam2_image.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        cap = cap_masks.get(fidx)
        neg_points = get_negative_points(rx, ry, cap, img_h, img_w, n_points=4,
                                         distance=CONFIG["negative_prompt_distance"])
        
        with torch.inference_mode():
            if neg_points:
                all_points = [(rx, ry)] + neg_points
                all_labels = [1] + [0] * len(neg_points)
                masks, scores, _ = sam2_image.predict(
                    point_coords=np.array(all_points),
                    point_labels=np.array(all_labels),
                    multimask_output=True
                )
            else:
                masks, scores, _ = sam2_image.predict(
                    point_coords=np.array([[rx, ry]]),
                    point_labels=np.array([1]),
                    multimask_output=True
                )
        mask = masks[np.argmax(scores)]
        
        centroid = get_mask_centroid(mask)
        if centroid is None:
            centroid = (rx, ry)
        
        all_masks[fidx][oid] = mask
        tracking[fidx][oid] = centroid
        new_adds.append((oid, fidx, (rx, ry)))
    
    def retrack_new():
        if not new_adds:
            return
        
        print("\n  Tracking new additions...")
        
        with torch.inference_mode():
            state = sam2_video.init_state(video_path=FRAME_DIR)
            
            for oid, fidx, (rx, ry) in new_adds:
                cap = cap_masks.get(fidx)
                neg_points = get_negative_points(rx, ry, cap, img_h, img_w, n_points=4,
                                                 distance=CONFIG["negative_prompt_distance"])
                
                if neg_points:
                    all_points = [(rx, ry)] + neg_points
                    all_labels = [1] + [0] * len(neg_points)
                    sam2_video.add_new_points_or_box(
                        state, frame_idx=fidx, obj_id=oid,
                        points=np.array(all_points, dtype=np.float32),
                        labels=np.array(all_labels, dtype=np.int32)
                    )
                else:
                    sam2_video.add_new_points_or_box(
                        state, frame_idx=fidx, obj_id=oid,
                        points=np.array([(rx, ry)], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32)
                    )
            
            for fidx, ids, logits in tqdm(sam2_video.propagate_in_video(state),
                                          total=len(frames), desc="  "):
                cap = cap_masks.get(fidx)
                cap_bool = cap > 0 if cap is not None else None
                
                for i, oid in enumerate(ids):
                    mask = (logits[i] > 0).cpu().numpy().squeeze()
                    
                    if cap_bool is not None and mask_inside_cap_ratio(mask, cap_bool) < CONFIG["min_cap_overlap"]:
                        continue
                    
                    centroid = get_mask_centroid(mask)
                    if centroid:
                        all_masks[fidx][int(oid)] = mask
                        tracking[fidx][int(oid)] = centroid
        
        new_adds.clear()
        print(" Done")
    
    def mouse(e, x, y, f, p):
        if e != cv2.EVENT_LBUTTONDOWN:
            return
        rx, ry = int(x/SCALE), int(y/SCALE)
        
        cap = cap_masks.get(idx)
        if cap is not None:
            cap_bool = cap > 0
            if not (0 <= ry < cap_bool.shape[0] and 0 <= rx < cap_bool.shape[1] and cap_bool[ry, rx]):
                print("  >>> Outside cap")
                return
        
        existing = list(tracking.get(idx, {}).values())
        for ex, ey in existing:
            if np.sqrt((rx-ex)**2 + (ry-ey)**2) < CONFIG["duplicate_radius"]:
                print("  >>> Too close to existing")
                return
        
        add_obj(rx, ry, idx, curr_lm < NUM_LANDMARKS)
    
    cv2.namedWindow("ADD LANDMARKS - Click NAS, LPA, RPA then SPACE")
    cv2.setMouseCallback("ADD LANDMARKS - Click NAS, LPA, RPA then SPACE", mouse)
    
    while True:
        if playing and time.time() - last_t > 1.0/CONFIG["playback_fps"]:
            idx = (idx + 1) % len(frames)
            last_t = time.time()
        
        img = cv2.imread(os.path.join(FRAME_DIR, frames[idx]))
        disp, _ = resize_for_display(img, CONFIG["display_height"])
        
        cap = cap_masks.get(idx)
        if cap is not None:
            capr = cv2.resize((cap > 0).astype(np.uint8), (disp.shape[1], disp.shape[0]))
            disp[capr == 0] = disp[capr == 0] // 2
        
        fm = all_masks.get(idx, {})
        if fm:
            disp = draw_all_masks(disp, fm)
        
        cv2.rectangle(disp, (0,0), (disp.shape[1], 95), (0,0,0), -1)
        
        st = "PLAY" if playing else "PAUSE"
        is_kf = "*KF" if idx in vggt_frames_set else ""
        cv2.putText(disp, f"Frame {idx}/{len(frames)-1} | {st} | Tracked: {len(fm)} {is_kf}", 
                   (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        if curr_lm < NUM_LANDMARKS:
            lms = " ".join([f"{LANDMARK_SHORT[i]}:{'OK' if lm_done[i] else '?'}" for i in range(3)])
            cv2.putText(disp, f"Click {LANDMARK_NAMES[curr_lm]} | {lms}", 
                       (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        else:
            cv2.putText(disp, "Landmarks OK | Click missed electrodes (optional)", 
                       (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        cv2.putText(disp, f"A/D:Nav W/S:+/-10 P:Play T:Track({len(new_adds)}) SPACE:Done", 
                   (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        cv2.imshow("ADD LANDMARKS - Click NAS, LPA, RPA then SPACE", disp)
        k = cv2.waitKey(1) & 0xFF
        
        if k in [ord('a'), 81] and not playing: idx = max(0, idx-1)
        elif k in [ord('d'), 83] and not playing: idx = min(len(frames)-1, idx+1)
        elif k == ord('w') and not playing: idx = max(0, idx-10)
        elif k == ord('s') and not playing: idx = min(len(frames)-1, idx+10)
        elif k == ord('p'): playing = not playing
        elif k == ord('t'): retrack_new()
        elif k == 32:
            if not all(lm_done):
                print("  >>> Add all 3 landmarks first! (NAS, LPA, RPA)")
                continue
            if new_adds:
                retrack_new()
            print("[OK] Landmarks added")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
        elif k == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
    
    return all_masks, tracking


# REVIEW


def review(frames, all_masks, cap_masks, vggt_data):
    print("REVIEW - Press SPACE to Accept, R to Redo")
    
    vggt_frames = set(vggt_data["s1_to_vggt"].keys())
    idx = 0
    playing = False
    last_t = time.time()
    
    cv2.namedWindow("REVIEW - SPACE:Accept R:Redo")
    
    while True:
        if playing and time.time() - last_t > 1.0/CONFIG["playback_fps"]:
            idx = (idx + 1) % len(frames)
            last_t = time.time()
        
        img = cv2.imread(os.path.join(FRAME_DIR, frames[idx]))
        disp, _ = resize_for_display(img, CONFIG["display_height"])
        
        cap = cap_masks.get(idx)
        if cap is not None:
            capr = cv2.resize((cap > 0).astype(np.uint8), (disp.shape[1], disp.shape[0]))
            disp[capr == 0] = disp[capr == 0] // 2
        
        fm = all_masks.get(idx, {})
        if fm:
            disp = draw_all_masks(disp, fm)
        
        cv2.rectangle(disp, (0,0), (disp.shape[1], 45), (0,0,0), -1)
        is_kf = "*KF" if idx in vggt_frames else ""
        cv2.putText(disp, f"Frame {idx} | Tracked: {len(fm)} {is_kf} | P:Play SPACE:Accept R:Redo",
                   (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        cv2.imshow("REVIEW - SPACE:Accept R:Redo", disp)
        k = cv2.waitKey(1) & 0xFF
        
        if k in [ord('a'), 81] and not playing: idx = max(0, idx-1)
        elif k in [ord('d'), 83] and not playing: idx = min(len(frames)-1, idx+1)
        elif k == ord('w') and not playing: idx = max(0, idx-10)
        elif k == ord('s') and not playing: idx = min(len(frames)-1, idx+10)
        elif k == ord('p'): playing = not playing
        elif k == 32:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("[OK] Tracking accepted!")
            return True
        elif k == ord('r'):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("  -> Redoing...")
            return False
        elif k == ord('q'):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            sys.exit(0)
    
    return True



# SAVE RESULTS


def save_results(tracking, all_masks):
    print("\n--- Saving Results ---")
    
    with open(TRACKING_FILE, "wb") as f:
        pickle.dump(tracking, f)
    print(f"  [OK] {TRACKING_FILE}")
    
    with open(MASKS_CACHE_FILE, "wb") as f:
        pickle.dump(all_masks, f)
    print(f"  [OK] {MASKS_CACHE_FILE}")
    
    # Count objects
    all_ids = set()
    for frame_data in tracking.values():
        all_ids.update(frame_data.keys())
    
    n_landmarks = sum(1 for i in all_ids if i < NUM_LANDMARKS)
    n_electrodes = sum(1 for i in all_ids if i >= ELECTRODE_START_ID)
    
    print(f"\n  Landmarks: {n_landmarks}")
    print(f"  Electrodes: {n_electrodes}")
    print(f"  Total frames: {len(tracking)}")



# MAIN


def main():
    video = select_video()
    
    print("EEG ELECTRODE REGISTRATION")
    print("VGGT-Assisted Tracking with SAM2 + YOLO")

    
    frames, crop_info = extract_frames(video)
    vggt_data = run_vggt(frames, crop_info)
    yolo, sam2_video, sam2_image = load_models()
    
    while True:
        cap_masks = create_cap_mask(frames, sam2_video, sam2_image)
        
        all_masks, tracking, electrode_ids = vggt_assisted_tracking(
            frames, cap_masks, yolo, sam2_video, sam2_image, vggt_data, crop_info)
        
        all_masks, tracking = interactive_add(
            frames, all_masks, tracking, cap_masks, sam2_video, sam2_image, 
            vggt_data, crop_info)
        
        if review(frames, all_masks, cap_masks, vggt_data):
            break
        
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
    
    save_results(tracking, all_masks)
    

    print("SCRIPT 1 COMPLETE!")
    print("Next: python script3.py")



if __name__ == "__main__":
    main()