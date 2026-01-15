"""
SCRIPT 1: ANNOTATION & TRACKING 

KEY INSIGHT: Head is STATIONARY, only camera moves.
             -> We can PROJECT 3D points to 2D instead of tracking in 2D.

APPROACH:
    1. Run VGGT FIRST to get camera poses for ALL frames
    2. User clicks landmarks/electrodes in MULTIPLE key frames (different views)
    3. Triangulate 3D positions immediately from multi-view clicks
    4. PROJECT 3D positions back to ALL frames
    5. Use SAM2 only for MASK REFINEMENT at projected locations (not tracking)

BENEFITS:
    - No tracking drift or loss
    - Consistent IDs across all frames
    - Robust to large viewpoint changes
    - Catches more electrodes (multi-view detection)

================================================================================
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


# ==============================================================================
# SAM2 DTYPE PATCH
# ==============================================================================

print("SCRIPT 1: ANNOTATION & TRACKING")

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

print("[OK] SAM2 dtype patch applied")



# CONFIGURATION


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
print(f"Device: {DEVICE.upper()}")

# Path configuration
# Script is at: .../scripts/script1.py
# Base dir is:  .../video-eeg-electrode-registration/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts
BASE_DIR = os.path.dirname(SCRIPT_DIR)                    # video-eeg-electrode-registration

VIDEO_DIR = os.path.join(BASE_DIR, "data", "Video_Recordings")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VGGT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "vggt_output")
TEMP_VGGT_DIR = os.path.join(BASE_DIR, "data", "vggt_ready_frames")
VGGT_REPO_PATH = os.path.join(SCRIPT_DIR, "vggt")  # scripts/vggt (VGGT repo inside scripts)

TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
CROP_INFO_FILE = os.path.join(RESULTS_DIR, "crop_info.json")
MASKS_CACHE_FILE = os.path.join(RESULTS_DIR, "masks_cache.pkl")
POINTS_3D_FILE = os.path.join(RESULTS_DIR, "points_3d_intermediate.pkl")

YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train4", "weights", "best.pt")
SAM2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam2_hiera_small.pt")
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
SAM2_CONFIG_NAME = "sam2_hiera_s.yaml"

# Print paths for debugging
print(f"Base directory: {BASE_DIR}")
print(f"Video directory: {VIDEO_DIR}")
print(f"Results directory: {RESULTS_DIR}")

SAM2_CONFIG = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2", SAM2_CONFIG_NAME)
if not os.path.exists(SAM2_CONFIG):
    for p in [os.path.join(BASE_DIR, "configs", "sam2", SAM2_CONFIG_NAME),
              os.path.join(BASE_DIR, "configs", SAM2_CONFIG_NAME)]:
        if os.path.exists(p): SAM2_CONFIG = p; break

# Object IDs
CAP_MASK_ID = 999
LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
NUM_LANDMARKS = 3
ELECTRODE_START_ID = 100

LANDMARK_NAMES = {0: "NAS (Nasion)", 1: "LPA (Left ear)", 2: "RPA (Right ear)"}
LANDMARK_SHORT = {0: "NAS", 1: "LPA", 2: "RPA"}
LANDMARK_COLORS = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}  # Red, Blue, Green

VGGT_SIZE = 518
MAX_VGGT_FRAMES = 35  # Reduced for laptop GPU memory - can increase if you have more VRAM

CONFIG = {
    "frame_skip": 5,  # Less skip = more frames = better reconstruction
    "display_height": 700,
    "yolo_conf": 0.25,
    "mask_alpha": 0.5,
    "playback_fps": 15,
    "cap_mask_expansion": 1.05,
    # Projection-based settings
    "projection_search_radius": 30,  # pixels - how far from projection to search
    "min_triangulation_angle": 5.0,  # degrees - minimum angle between views
    "sam_box_size": 20,  # Small box for SAM refinement
    "visibility_depth_threshold": 0.1,  # Relative depth threshold for visibility
}




# COLORS & HELPERS

def get_color(obj_id):
    if obj_id in LANDMARK_COLORS:
        return LANDMARK_COLORS[obj_id]
    if obj_id == CAP_MASK_ID:
        return (0, 255, 255)
    i = obj_id - ELECTRODE_START_ID if obj_id >= ELECTRODE_START_ID else obj_id
    hue = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return (int(b * 255), int(g * 255), int(r * 255))


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



# COORDINATE TRANSFORMATIONS

def script1_to_vggt_coords(u, v, crop_w, crop_h, vggt_size=518):
    """Convert Script1 pixel coords to VGGT coords."""
    scale = vggt_size / max(crop_w, crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    pad_w = (vggt_size - new_w) // 2
    pad_h = (vggt_size - new_h) // 2
    return u * scale + pad_w, v * scale + pad_h


def vggt_to_script1_coords(u_vggt, v_vggt, crop_w, crop_h, vggt_size=518):
    """Convert VGGT coords back to Script1 pixel coords."""
    scale = vggt_size / max(crop_w, crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    pad_w = (vggt_size - new_w) // 2
    pad_h = (vggt_size - new_h) // 2
    return (u_vggt - pad_w) / scale, (v_vggt - pad_h) / scale


def unproject_pixel(u, v, depth_map, intrinsic, extrinsic):
    """Unproject 2D pixel to 3D world coordinates."""
    H, W = depth_map.shape
    
    if not (0 <= u < W - 1 and 0 <= v < H - 1):
        return None
    
    # Bilinear interpolation for depth
    u0, v0 = int(u), int(v)
    u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)
    du, dv = u - u0, v - v0
    
    z = (depth_map[v0, u0] * (1 - du) * (1 - dv) +
         depth_map[v0, u1] * du * (1 - dv) +
         depth_map[v1, u0] * (1 - du) * dv +
         depth_map[v1, u1] * du * dv)
    
    if z <= 0 or not np.isfinite(z):
        return None
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    
    P_cam = np.array([x_cam, y_cam, z, 1.0])
    P_world = (np.linalg.inv(extrinsic) @ P_cam)[:3]
    
    return P_world


def project_3d_to_2d(point_3d, intrinsic, extrinsic):
    """Project 3D world point to 2D pixel coordinates."""
    # World to camera
    P_world = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    P_cam = extrinsic @ P_world
    
    x_cam, y_cam, z_cam = P_cam[0], P_cam[1], P_cam[2]
    
    if z_cam <= 0:
        return None, False  # Behind camera
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    
    return (u, v), True


def check_visibility(point_3d, frame_idx, depths, intrinsics, extrinsics, crop_info, threshold=0.1):
    """Check if a 3D point is visible in a frame (not occluded)."""
    proj, in_front = project_3d_to_2d(point_3d, intrinsics[frame_idx], extrinsics[frame_idx])
    
    if not in_front:
        return False, None
    
    u_vggt, v_vggt = proj
    
    # Check bounds
    H, W = depths[frame_idx].shape
    if not (0 <= u_vggt < W and 0 <= v_vggt < H):
        return False, None
    
    # Get expected depth (distance to camera)
    P_world = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    P_cam = extrinsics[frame_idx] @ P_world
    expected_depth = P_cam[2]
    
    # Get actual depth from depth map
    actual_depth = depths[frame_idx][int(v_vggt), int(u_vggt)]
    
    # Point is visible if depth matches (within threshold)
    if actual_depth > 0:
        relative_diff = abs(expected_depth - actual_depth) / expected_depth
        visible = relative_diff < threshold
    else:
        visible = False
    
    # Convert to script1 coords
    u_s1, v_s1 = vggt_to_script1_coords(u_vggt, v_vggt, crop_info["w"], crop_info["h"])
    
    return visible, (u_s1, v_s1)



# VIDEO/FRAME FUNCTIONS


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
    
    print(f"[OK] Extracted {len(frames)} frames")
    return frames, crop_info



# VGGT - RUN FIRST!

def run_vggt_full(frames, crop_info):
    """Run VGGT on ALL frames (or as many as possible) for complete camera poses."""

    print("RUNNING VGGT FOR CAMERA POSES")

    
    # Clear memory before starting
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Print available memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        used_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU Memory: {used_mem:.1f}GB used / {total_mem:.1f}GB total")
    
    sys.path.insert(0, VGGT_REPO_PATH)
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
    os.makedirs(VGGT_OUTPUT_DIR, exist_ok=True)
    
    all_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
    
    # Use as many frames as possible (up to MAX_VGGT_FRAMES)
    n = min(len(all_paths), MAX_VGGT_FRAMES)
    indices = np.linspace(0, len(all_paths)-1, n, dtype=int)
    selected = [all_paths[i] for i in indices]
    s1_indices = [int(os.path.basename(p).split('.')[0]) for p in selected]
    
    print(f"  Processing {n} frames for VGGT...")
    
    if os.path.exists(TEMP_VGGT_DIR): 
        shutil.rmtree(TEMP_VGGT_DIR)
    os.makedirs(TEMP_VGGT_DIR)
    
    # Prepare frames for VGGT
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
    
    # Clear memory before loading model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Load VGGT model
    checkpoint = os.path.join(CHECKPOINT_DIR, "vggt_model.pt")
    model = VGGT()
    if not os.path.exists(checkpoint):
        print("  Downloading VGGT weights...")
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location="cpu", progress=True)  # Load to CPU first
        torch.save(state, checkpoint)
        del state
        gc.collect()
    
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))  # Load to CPU first
    model.to(DEVICE).eval()
    
    # Clear memory after model load
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    print("  Loading and preprocessing images...")
    images = load_and_preprocess_images(processed).to(DEVICE)
    
    print(f"  Image tensor shape: {images.shape}")
    print(f"  Running VGGT inference (this may take a minute)...")
    
    # Use autocast for memory efficiency
    try:
        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):  # Use float16 instead of bfloat16
                    preds = model(images)
            else:
                preds = model(images)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n  ERROR: GPU out of memory!")
            print("  Try reducing MAX_VGGT_FRAMES in the script (current: {})".format(MAX_VGGT_FRAMES))
            print("  Or close other applications using GPU memory.")
            
            # Clean up
            del model, images
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            
            sys.exit(1)
        else:
            raise e
    
    print("  Processing VGGT outputs...")
    
    E, K = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    depth = preds["depth"].float().cpu().numpy().squeeze()
    E = E.float().cpu().numpy().squeeze()
    K = K.float().cpu().numpy().squeeze()
    
    # Free GPU memory immediately
    del preds, images, model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Ensure 4x4 extrinsics
    if E.shape[-2:] == (3,4):
        E4 = np.zeros((len(s1_indices), 4, 4))
        for i in range(len(s1_indices)):
            E4[i] = np.eye(4)
            E4[i,:3,:] = E[i]
        E = E4
    
    # Save results
    np.savez_compressed(os.path.join(VGGT_OUTPUT_DIR, "reconstruction.npz"),
                       depth=depth, extrinsics=E, intrinsics=K,
                       frame_mapping_keys=np.arange(len(s1_indices)),
                       frame_mapping_values=np.array(s1_indices))
    
    print(f"VGGT complete - {len(s1_indices)} frames processed")
    
    # Build mapping for quick lookup
    vggt_data = {
        "depths": depth,
        "extrinsics": E,
        "intrinsics": K,
        "s1_to_vggt": {int(s1): int(vggt) for vggt, s1 in enumerate(s1_indices)},
        "vggt_to_s1": {int(vggt): int(s1) for vggt, s1 in enumerate(s1_indices)},
        "s1_indices": s1_indices,
    }
    
    return vggt_data



# CAP MASK


def load_models():
    print(f"\nLoading models...")
    
    yolo = None
    if os.path.exists(YOLO_WEIGHTS):
        yolo = YOLO(YOLO_WEIGHTS)
    else:
        print("  WARNING: YOLO not found, will use manual detection only")
    
    if not os.path.exists(SAM2_CHECKPOINT):
        print("Downloading SAM2...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        urllib.request.urlretrieve(SAM2_URL, SAM2_CHECKPOINT)
    
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_image = SAM2ImagePredictor(sam2_model.float().eval())
    
    print("[OK] Models loaded")
    return yolo, sam2_image


def create_cap_mask_single_frame(frame_path, sam2_image):
    """Create cap mask for a single frame interactively."""
    img = cv2.imread(frame_path)
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
        
        if k == ord('y') and cap is not None: 
            break
        elif k == ord('r'): 
            cap = None
        elif k == ord('q'): 
            sys.exit(0)
    
    cv2.destroyWindow("Cap")
    return cap



# MULTI-VIEW ANNOTATION INTERFACE


class MultiViewAnnotator:
    """
    Annotate landmarks and electrodes across multiple views.
    Triangulates 3D positions from multi-view observations.
    """
    
    def __init__(self, frames, vggt_data, crop_info, sam2_image):
        self.frames = frames
        self.vggt_data = vggt_data
        self.crop_info = crop_info
        self.sam2_image = sam2_image
        
        # Annotations: obj_id -> {frame_idx: (u, v)}
        self.annotations_2d = {}
        
        # 3D positions: obj_id -> np.array([x, y, z])
        self.positions_3d = {}
        
        # Cap masks per frame
        self.cap_masks = {}
        
        # For display
        img0 = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
        _, self.scale = resize_for_display(img0, CONFIG["display_height"])
        self.img_h, self.img_w = img0.shape[:2]
        
        # Track current state
        self.current_obj_id = None
        self.next_electrode_id = ELECTRODE_START_ID
    
    def get_vggt_frame_idx(self, s1_idx):
        """Get VGGT frame index for a script1 frame index."""
        return self.vggt_data["s1_to_vggt"].get(s1_idx, None)
    
    def unproject_click(self, u, v, s1_idx):
        """Unproject a 2D click to 3D."""
        vggt_idx = self.get_vggt_frame_idx(s1_idx)
        if vggt_idx is None:
            return None
        
        u_vggt, v_vggt = script1_to_vggt_coords(u, v, self.crop_info["w"], self.crop_info["h"])
        
        return unproject_pixel(
            u_vggt, v_vggt,
            self.vggt_data["depths"][vggt_idx],
            self.vggt_data["intrinsics"][vggt_idx],
            self.vggt_data["extrinsics"][vggt_idx]
        )
    
    def triangulate_point(self, obj_id):
        """
        Triangulate 3D position from all observations of an object.
        Uses WEIGHTED averaging based on:
          1. View angle (prefer head-on views)
          2. Depth confidence (prefer closer points)
          3. Multi-view agreement
        """
        if obj_id not in self.annotations_2d:
            return None
        
        observations = self.annotations_2d[obj_id]
        if len(observations) < 1:
            return None
        
        # Collect 3D points with quality weights
        points_3d = []
        weights = []
        
        for s1_idx, (u, v) in observations.items():
            vggt_idx = self.get_vggt_frame_idx(s1_idx)
            if vggt_idx is None:
                continue
            
            p3d = self.unproject_click(u, v, s1_idx)
            if p3d is None:
                continue
            
            # Compute quality weight
            weight = 1.0
            
            # 1. Depth-based weight: closer points are more reliable
            u_vggt, v_vggt = script1_to_vggt_coords(u, v, self.crop_info["w"], self.crop_info["h"])
            depth_map = self.vggt_data["depths"][vggt_idx]
            H, W = depth_map.shape
            if 0 <= int(v_vggt) < H and 0 <= int(u_vggt) < W:
                depth = depth_map[int(v_vggt), int(u_vggt)]
                if depth > 0:
                    # Inverse depth weighting (closer = higher weight)
                    weight *= 1.0 / (1.0 + depth)
            
            # 2. View angle weight: prefer when point is near image center
            # (less distortion, better depth estimation)
            img_center_u, img_center_v = W / 2, H / 2
            dist_from_center = np.sqrt((u_vggt - img_center_u)**2 + (v_vggt - img_center_v)**2)
            max_dist = np.sqrt(img_center_u**2 + img_center_v**2)
            center_weight = 1.0 - 0.5 * (dist_from_center / max_dist)  # 0.5 to 1.0
            weight *= center_weight
            
            points_3d.append(p3d)
            weights.append(weight)
        
        if not points_3d:
            return None
        
        points_3d = np.array(points_3d)
        weights = np.array(weights)
        
        # Outlier removal if multiple observations
        if len(points_3d) >= 3:
            # First pass: compute median
            median = np.median(points_3d, axis=0)
            dists = np.linalg.norm(points_3d - median, axis=1)
            
            # Adaptive threshold
            threshold = np.mean(dists) + 2 * np.std(dists) if np.std(dists) > 0 else np.inf
            mask = dists < threshold
            
            if np.sum(mask) >= 1:
                # Weighted average of inliers
                valid_points = points_3d[mask]
                valid_weights = weights[mask]
                valid_weights = valid_weights / valid_weights.sum()  # Normalize
                return np.average(valid_points, axis=0, weights=valid_weights)
        
        # Weighted average
        weights = weights / weights.sum()
        return np.average(points_3d, axis=0, weights=weights)
    
    def project_to_frame(self, obj_id, s1_idx):
        """Project a 3D point to a specific frame."""
        if obj_id not in self.positions_3d:
            return None, False
        
        vggt_idx = self.get_vggt_frame_idx(s1_idx)
        if vggt_idx is None:
            return None, False
        
        point_3d = self.positions_3d[obj_id]
        
        proj, in_front = project_3d_to_2d(
            point_3d,
            self.vggt_data["intrinsics"][vggt_idx],
            self.vggt_data["extrinsics"][vggt_idx]
        )
        
        if not in_front:
            return None, False
        
        u_vggt, v_vggt = proj
        u_s1, v_s1 = vggt_to_script1_coords(u_vggt, v_vggt, self.crop_info["w"], self.crop_info["h"])
        
        # Check if in bounds
        if not (0 <= u_s1 < self.img_w and 0 <= v_s1 < self.img_h):
            return None, False
        
        return (u_s1, v_s1), True
    
    def update_3d_positions(self):
        """Update all 3D positions from current annotations."""
        for obj_id in self.annotations_2d:
            p3d = self.triangulate_point(obj_id)
            if p3d is not None:
                self.positions_3d[obj_id] = p3d
    
    def annotate_landmarks(self):
        """Interactive landmark annotation across multiple frames."""
        print("LANDMARK ANNOTATION")
        print("Navigate to frames where each landmark is clearly visible")
        print("Click to mark, annotations from multiple views improve accuracy")
        
        for landmark_id, name in [(LANDMARK_NAS, "NAS (Nasion)"), 
                                   (LANDMARK_LPA, "LPA (Left ear pin)"),
                                   (LANDMARK_RPA, "RPA (Right ear pin)")]:
            self._annotate_single_object(landmark_id, name, is_landmark=True)
        
        self.update_3d_positions()
        
        print("\n[OK] Landmarks annotated:")
        for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
            n_obs = len(self.annotations_2d.get(lid, {}))
            has_3d = lid in self.positions_3d
            print(f"  {LANDMARK_SHORT[lid]}: {n_obs} observations, 3D: {'Yes' if has_3d else 'No'}")
    
    def annotate_electrodes(self, yolo=None):
        """Interactive electrode annotation."""
        print("ELECTRODE ANNOTATION")
        print("Navigate through frames and click on electrodes")
        print("SPACE: Auto-detect with YOLO | Click: Add electrode | Right-click: Remove")
        print("Y: Done with this view | N: Next electrode | Q: Finish all")
        
        # Select key frames (frames with VGGT data)
        vggt_frames = sorted(self.vggt_data["s1_to_vggt"].keys())
        
        idx = 0
        
        cv2.namedWindow("Electrodes")
        
        click_pos = None
        
        def mouse(e, x, y, f, p):
            nonlocal click_pos
            if e == cv2.EVENT_LBUTTONDOWN:
                click_pos = (int(x / self.scale), int(y / self.scale))
        
        cv2.setMouseCallback("Electrodes", mouse)
        
        while True:
            s1_idx = vggt_frames[idx]
            frame_path = os.path.join(FRAME_DIR, self.frames[s1_idx])
            img = cv2.imread(frame_path)
            disp, _ = resize_for_display(img, CONFIG["display_height"])
            
            # Draw existing annotations
            for obj_id in self.annotations_2d:
                if obj_id < ELECTRODE_START_ID:
                    continue  # Skip landmarks
                
                color = get_color(obj_id)
                
                # Draw from 2D annotation if this frame has one
                if s1_idx in self.annotations_2d[obj_id]:
                    u, v = self.annotations_2d[obj_id][s1_idx]
                    dx, dy = int(u * self.scale), int(v * self.scale)
                    cv2.circle(disp, (dx, dy), 8, color, -1)
                    cv2.circle(disp, (dx, dy), 10, (255, 255, 255), 2)
                    label = f"E{obj_id - ELECTRODE_START_ID}"
                    cv2.putText(disp, label, (dx + 12, dy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Or draw projection from 3D
                elif obj_id in self.positions_3d:
                    proj, visible = self.project_to_frame(obj_id, s1_idx)
                    if visible and proj:
                        dx, dy = int(proj[0] * self.scale), int(proj[1] * self.scale)
                        cv2.circle(disp, (dx, dy), 6, color, 1)  # Hollow = projected
                        label = f"E{obj_id - ELECTRODE_START_ID}*"
                        cv2.putText(disp, label, (dx + 12, dy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            # Draw landmarks
            for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
                if lid in self.positions_3d:
                    proj, visible = self.project_to_frame(lid, s1_idx)
                    if visible and proj:
                        dx, dy = int(proj[0] * self.scale), int(proj[1] * self.scale)
                        cv2.circle(disp, (dx, dy), 5, LANDMARK_COLORS[lid], -1)
                        cv2.putText(disp, LANDMARK_SHORT[lid], (dx + 8, dy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, LANDMARK_COLORS[lid], 1)
            
            # UI
            n_electrodes = len([k for k in self.annotations_2d if k >= ELECTRODE_START_ID])
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(disp, f"Frame {s1_idx} ({idx+1}/{len(vggt_frames)}) | Electrodes: {n_electrodes}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(disp, "A/D:Nav | Click:Add | SPACE:AutoDetect | Q:Done", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            cv2.imshow("Electrodes", disp)
            k = cv2.waitKey(1) & 0xFF
            
            # Navigation
            if k in [ord('a'), 81]: 
                idx = max(0, idx - 1)
            elif k in [ord('d'), 83]: 
                idx = min(len(vggt_frames) - 1, idx + 1)
            elif k == ord('w'): 
                idx = max(0, idx - 5)
            elif k == ord('s'): 
                idx = min(len(vggt_frames) - 1, idx + 5)
            
            # Add electrode from click
            elif click_pos is not None:
                u, v = click_pos
                click_pos = None
                
                # Check if clicking near existing electrode (to select/adjust)
                clicked_existing = False
                for obj_id in list(self.annotations_2d.keys()):
                    if obj_id < ELECTRODE_START_ID:
                        continue
                    
                    # Check 2D annotation
                    if s1_idx in self.annotations_2d[obj_id]:
                        eu, ev = self.annotations_2d[obj_id][s1_idx]
                        if np.sqrt((u - eu)**2 + (v - ev)**2) < 20:
                            # Remove this observation
                            del self.annotations_2d[obj_id][s1_idx]
                            if len(self.annotations_2d[obj_id]) == 0:
                                del self.annotations_2d[obj_id]
                                if obj_id in self.positions_3d:
                                    del self.positions_3d[obj_id]
                            clicked_existing = True
                            print(f"  Removed E{obj_id - ELECTRODE_START_ID} from frame {s1_idx}")
                            break
                    
                    # Check projection
                    elif obj_id in self.positions_3d:
                        proj, visible = self.project_to_frame(obj_id, s1_idx)
                        if visible and proj:
                            if np.sqrt((u - proj[0])**2 + (v - proj[1])**2) < 20:
                                # Add observation to existing electrode
                                self.annotations_2d[obj_id][s1_idx] = (u, v)
                                clicked_existing = True
                                print(f"  Added observation to E{obj_id - ELECTRODE_START_ID}")
                                break
                
                if not clicked_existing:
                    # Create new electrode
                    new_id = self.next_electrode_id
                    self.next_electrode_id += 1
                    self.annotations_2d[new_id] = {s1_idx: (u, v)}
                    print(f"  Added E{new_id - ELECTRODE_START_ID} at frame {s1_idx}")
                
                # Update 3D
                self.update_3d_positions()
            
            # Auto-detect with YOLO
            elif k == 32 and yolo is not None:
                res = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
                if res and res[0].boxes is not None:
                    added = 0
                    for box in res[0].boxes.xyxy.cpu().numpy():
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        
                        # Check if near existing
                        is_dup = False
                        for obj_id in self.annotations_2d:
                            if obj_id < ELECTRODE_START_ID:
                                continue
                            if s1_idx in self.annotations_2d[obj_id]:
                                eu, ev = self.annotations_2d[obj_id][s1_idx]
                                if np.sqrt((cx - eu)**2 + (cy - ev)**2) < 25:
                                    is_dup = True
                                    break
                            elif obj_id in self.positions_3d:
                                proj, _ = self.project_to_frame(obj_id, s1_idx)
                                if proj and np.sqrt((cx - proj[0])**2 + (cy - proj[1])**2) < 25:
                                    is_dup = True
                                    break
                        
                        if not is_dup:
                            new_id = self.next_electrode_id
                            self.next_electrode_id += 1
                            self.annotations_2d[new_id] = {s1_idx: (float(cx), float(cy))}
                            added += 1
                    
                    print(f"  Auto-detected {added} new electrodes")
                    self.update_3d_positions()
            
            # Done
            elif k == ord('q'):
                break
        
        cv2.destroyWindow("Electrodes")
        
        # Final 3D update
        self.update_3d_positions()
        
        n_electrodes = len([k for k in self.positions_3d if k >= ELECTRODE_START_ID])
        print(f"\n[OK] {n_electrodes} electrodes with 3D positions")
    
    def _annotate_single_object(self, obj_id, name, is_landmark=True):
        """Annotate a single object across multiple views."""
        print(f"\n  --- Annotate {name} ---")
        print(f"  Click on {name} in multiple frames for better accuracy")
        print(f"  Y: Done | R: Clear all | A/D: Navigate")
        
        self.annotations_2d[obj_id] = {}
        
        vggt_frames = sorted(self.vggt_data["s1_to_vggt"].keys())
        idx = len(vggt_frames) // 2  # Start in middle
        
        color = LANDMARK_COLORS.get(obj_id, (0, 255, 255))
        
        cv2.namedWindow(f"Annotate {name}")
        
        click_pos = None
        
        def mouse(e, x, y, f, p):
            nonlocal click_pos
            if e == cv2.EVENT_LBUTTONDOWN:
                click_pos = (int(x / self.scale), int(y / self.scale))
        
        cv2.setMouseCallback(f"Annotate {name}", mouse)
        
        while True:
            s1_idx = vggt_frames[idx]
            frame_path = os.path.join(FRAME_DIR, self.frames[s1_idx])
            img = cv2.imread(frame_path)
            disp, _ = resize_for_display(img, CONFIG["display_height"])
            
            # Draw existing annotations for this object
            for ann_frame, (u, v) in self.annotations_2d[obj_id].items():
                if ann_frame == s1_idx:
                    # Current frame annotation
                    dx, dy = int(u * self.scale), int(v * self.scale)
                    cv2.circle(disp, (dx, dy), 10, color, -1)
                    cv2.circle(disp, (dx, dy), 12, (255, 255, 255), 2)
            
            # Draw projection if we have 3D
            if len(self.annotations_2d[obj_id]) >= 1:
                p3d = self.triangulate_point(obj_id)
                if p3d is not None:
                    vggt_idx = self.get_vggt_frame_idx(s1_idx)
                    if vggt_idx is not None:
                        proj, in_front = project_3d_to_2d(
                            p3d,
                            self.vggt_data["intrinsics"][vggt_idx],
                            self.vggt_data["extrinsics"][vggt_idx]
                        )
                        if in_front:
                            u_s1, v_s1 = vggt_to_script1_coords(proj[0], proj[1], 
                                                                  self.crop_info["w"], self.crop_info["h"])
                            dx, dy = int(u_s1 * self.scale), int(v_s1 * self.scale)
                            cv2.drawMarker(disp, (dx, dy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            # UI
            n_obs = len(self.annotations_2d[obj_id])
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(disp, f"{name} | Frame {s1_idx} | Observations: {n_obs}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(disp, "Click to mark | A/D:Nav | Y:Done | R:Clear", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow(f"Annotate {name}", disp)
            k = cv2.waitKey(1) & 0xFF
            
            if k in [ord('a'), 81]: 
                idx = max(0, idx - 1)
            elif k in [ord('d'), 83]: 
                idx = min(len(vggt_frames) - 1, idx + 1)
            elif k == ord('w'): 
                idx = max(0, idx - 5)
            elif k == ord('s'): 
                idx = min(len(vggt_frames) - 1, idx + 5)
            
            elif click_pos is not None:
                u, v = click_pos
                click_pos = None
                
                # Check if clicking on existing annotation (to remove)
                removed = False
                if s1_idx in self.annotations_2d[obj_id]:
                    eu, ev = self.annotations_2d[obj_id][s1_idx]
                    if np.sqrt((u - eu)**2 + (v - ev)**2) < 20:
                        del self.annotations_2d[obj_id][s1_idx]
                        removed = True
                        print(f"    Removed annotation from frame {s1_idx}")
                
                if not removed:
                    self.annotations_2d[obj_id][s1_idx] = (u, v)
                    print(f"    Added annotation at frame {s1_idx}: ({u:.0f}, {v:.0f})")
            
            elif k == ord('r'):
                self.annotations_2d[obj_id] = {}
                print(f"    Cleared all annotations for {name}")
            
            elif k == ord('y'):
                if len(self.annotations_2d[obj_id]) >= 1:
                    break
                else:
                    print(f"    Need at least 1 annotation!")
            
            elif k == ord('q'):
                sys.exit(0)
        
        cv2.destroyWindow(f"Annotate {name}")



# PROJECTION-BASED TRACKING

def generate_tracking_from_3d(frames, vggt_data, crop_info, positions_3d, sam2_image):
    """
    Generate 2D tracking data by projecting 3D positions to each frame.
    Optionally refine with SAM2.
    """

    print("GENERATING TRACKING FROM 3D PROJECTIONS")

    
    tracking = {}
    masks_cache = {}
    
    img0 = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
    img_h, img_w = img0.shape[:2]
    
    for s1_idx in tqdm(range(len(frames)), desc="Projecting"):
        vggt_idx = vggt_data["s1_to_vggt"].get(s1_idx, None)
        
        if vggt_idx is None:
            # Interpolate for frames without VGGT data
            # Find nearest frames with VGGT data
            s1_indices = sorted(vggt_data["s1_to_vggt"].keys())
            if not s1_indices:
                continue
            
            # Find bracketing frames
            before = [i for i in s1_indices if i < s1_idx]
            after = [i for i in s1_indices if i > s1_idx]
            
            if before and after:
                # Interpolate
                b_idx = before[-1]
                a_idx = after[0]
                t = (s1_idx - b_idx) / (a_idx - b_idx)
                
                b_vggt = vggt_data["s1_to_vggt"][b_idx]
                a_vggt = vggt_data["s1_to_vggt"][a_idx]
                
                # Simple linear interpolation of extrinsics (not ideal but workable)
                intrinsic = vggt_data["intrinsics"][b_vggt]  # Assume constant
                extrinsic = (1 - t) * vggt_data["extrinsics"][b_vggt] + t * vggt_data["extrinsics"][a_vggt]
            elif before:
                b_vggt = vggt_data["s1_to_vggt"][before[-1]]
                intrinsic = vggt_data["intrinsics"][b_vggt]
                extrinsic = vggt_data["extrinsics"][b_vggt]
            elif after:
                a_vggt = vggt_data["s1_to_vggt"][after[0]]
                intrinsic = vggt_data["intrinsics"][a_vggt]
                extrinsic = vggt_data["extrinsics"][a_vggt]
            else:
                continue
        else:
            intrinsic = vggt_data["intrinsics"][vggt_idx]
            extrinsic = vggt_data["extrinsics"][vggt_idx]
        
        frame_tracking = {}
        frame_masks = {}
        
        for obj_id, pos_3d in positions_3d.items():
            # Project to 2D
            proj, in_front = project_3d_to_2d(pos_3d, intrinsic, extrinsic)
            
            if not in_front:
                continue
            
            u_vggt, v_vggt = proj
            u_s1, v_s1 = vggt_to_script1_coords(u_vggt, v_vggt, crop_info["w"], crop_info["h"])
            
            # Check bounds
            if not (0 <= u_s1 < img_w and 0 <= v_s1 < img_h):
                continue
            
            # Simple visibility check using depth (if available)
            if vggt_idx is not None:
                depth_map = vggt_data["depths"][vggt_idx]
                H, W = depth_map.shape
                
                if 0 <= int(v_vggt) < H and 0 <= int(u_vggt) < W:
                    # Check if point is roughly at the expected depth
                    P_world = np.array([pos_3d[0], pos_3d[1], pos_3d[2], 1.0])
                    P_cam = extrinsic @ P_world
                    expected_depth = P_cam[2]
                    actual_depth = depth_map[int(v_vggt), int(u_vggt)]
                    
                    if actual_depth > 0:
                        relative_diff = abs(expected_depth - actual_depth) / max(expected_depth, 0.001)
                        if relative_diff > 0.3:  # More than 30% difference = likely occluded
                            continue
            
            frame_tracking[obj_id] = (u_s1, v_s1)
        
        if frame_tracking:
            tracking[s1_idx] = frame_tracking
            masks_cache[s1_idx] = frame_masks
    
    print(f" Generated tracking for {len(tracking)} frames")
    
    # Statistics
    all_obj_ids = set()
    for frame_data in tracking.values():
        all_obj_ids.update(frame_data.keys())
    
    n_landmarks = sum(1 for i in all_obj_ids if isinstance(i, int) and i < NUM_LANDMARKS)
    n_electrodes = sum(1 for i in all_obj_ids if isinstance(i, int) and i >= ELECTRODE_START_ID)
    
    print(f"  Landmarks: {n_landmarks}")
    print(f"  Electrodes: {n_electrodes}")
    
    return tracking, masks_cache



# REVIEW

def review_tracking(frames, tracking, positions_3d, vggt_data, crop_info):
    """Review the projected tracking results."""

    print("REVIEW TRACKING")
    
    idx = 0
    playing = False
    last_t = time.time()
    
    img0 = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
    _, scale = resize_for_display(img0, CONFIG["display_height"])
    
    cv2.namedWindow("Review")
    
    while True:
        if playing and time.time() - last_t > 1.0 / CONFIG["playback_fps"]:
            idx = (idx + 1) % len(frames)
            last_t = time.time()
        
        img = cv2.imread(os.path.join(FRAME_DIR, frames[idx]))
        disp, _ = resize_for_display(img, CONFIG["display_height"])
        
        # Draw tracking
        if idx in tracking:
            for obj_id, (u, v) in tracking[idx].items():
                color = get_color(obj_id)
                dx, dy = int(u * scale), int(v * scale)
                
                cv2.circle(disp, (dx, dy), 6, color, -1)
                cv2.circle(disp, (dx, dy), 8, (255, 255, 255), 1)
                
                if obj_id < NUM_LANDMARKS:
                    label = LANDMARK_SHORT.get(obj_id, f"L{obj_id}")
                else:
                    label = f"E{obj_id - ELECTRODE_START_ID}"
                
                cv2.putText(disp, label, (dx + 10, dy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # UI
        n_tracked = len(tracking.get(idx, {}))
        has_vggt = idx in vggt_data["s1_to_vggt"]
        
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 50), (0, 0, 0), -1)
        status = "PLAY" if playing else "PAUSE"
        vggt_status = "VGGT" if has_vggt else "interp"
        cv2.putText(disp, f"Frame {idx}/{len(frames)-1} | {status} | Tracked: {n_tracked} | {vggt_status}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp, "A/D:Nav | P:Play | SPACE:Accept | R:Redo", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Review", disp)
        k = cv2.waitKey(1) & 0xFF
        
        if k in [ord('a'), 81] and not playing: idx = max(0, idx - 1)
        elif k in [ord('d'), 83] and not playing: idx = min(len(frames) - 1, idx + 1)
        elif k == ord('w') and not playing: idx = max(0, idx - 10)
        elif k == ord('s') and not playing: idx = min(len(frames) - 1, idx + 10)
        elif k == ord('p'): playing = not playing
        elif k == 32:  # Space - accept
            cv2.destroyWindow("Review")
            return True
        elif k == ord('r'):  # Redo
            cv2.destroyWindow("Review")
            return False
        elif k == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)



# SAVE


def save_results(tracking, masks_cache, positions_3d):
    print("\n--- Saving Results ---")
    
    with open(TRACKING_FILE, "wb") as f:
        pickle.dump(tracking, f)
    print(f"  [OK] {TRACKING_FILE}")
    
    with open(MASKS_CACHE_FILE, "wb") as f:
        pickle.dump(masks_cache, f)
    print(f"  [OK] {MASKS_CACHE_FILE}")
    
    with open(POINTS_3D_FILE, "wb") as f:
        pickle.dump(positions_3d, f)
    print(f"  [OK] {POINTS_3D_FILE}")
    
    # Statistics
    all_ids = set()
    for frame_data in tracking.values():
        all_ids.update(frame_data.keys())
    
    n_landmarks = sum(1 for i in all_ids if isinstance(i, int) and i < NUM_LANDMARKS)
    n_electrodes = sum(1 for i in all_ids if isinstance(i, int) and i >= ELECTRODE_START_ID)
    
    print(f"\n  Landmarks: {n_landmarks}")
    print(f"  Electrodes: {n_electrodes}")
    print(f"  Total 3D points: {len(positions_3d)}")



# MAIN


def main():
    video = select_video()
    

    print("EEG ELECTRODE REGISTRATION")
    print("Projection-Based Tracking")

    
    # Step 1: Extract frames
    frames, crop_info = extract_frames(video)
    
    # Step 2: Run VGGT FIRST (for all camera poses)
    vggt_data = run_vggt_full(frames, crop_info)
    
    # Step 3: Load models
    yolo, sam2_image = load_models()
    
    while True:
        # Step 4: Multi-view annotation
        annotator = MultiViewAnnotator(frames, vggt_data, crop_info, sam2_image)
        
        # Annotate landmarks
        annotator.annotate_landmarks()
        
        # Annotate electrodes
        annotator.annotate_electrodes(yolo)
        
        # Step 5: Generate tracking from 3D projections
        tracking, masks_cache = generate_tracking_from_3d(
            frames, vggt_data, crop_info, annotator.positions_3d, sam2_image
        )
        
        # Step 6: Review
        if review_tracking(frames, tracking, annotator.positions_3d, vggt_data, crop_info):
            break
        
        print("\n  Restarting annotation...\n")
    
    # Step 7: Save
    save_results(tracking, masks_cache, annotator.positions_3d)
    

    print("SCRIPT 1 COMPLETE!")
    print("Next: python script2.py")



if __name__ == "__main__":
    main()