"""

SCRIPT 2: VGGT 3D RECONSTRUCTION


PURPOSE:
    Generate 3D depth maps and camera parameters from video frames.
    
INPUT:
    - frames/ from Script 1
    - crop_info.json from Script 1
    
OUTPUT:
    - vggt_output/reconstruction.npz containing:
        - depth maps (H x W per frame)
        - intrinsic matrices (camera internal parameters)
        - extrinsic matrices (camera pose per frame)
        - frame mapping (VGGT index ↔ Script 1 index)

WHAT VGGT DOES:
    Given multiple images, VGGT estimates:
    - Depth: How far each pixel is from the camera
    - Camera pose: Where the camera was for each image
    
    All frames are placed in a common 3D "world" coordinate system.

"""

import os
import sys
import glob
import time
import shutil
import numpy as np
import torch
import cv2
from tqdm import tqdm
from contextlib import nullcontext


# CONFIGURATION


print("=" * 70)
print("SCRIPT 2: VGGT 3D RECONSTRUCTION")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = os.path.join(BASE_DIR, "vggt")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "vggt_output")

FRAME_DIR = os.path.join(BASE_DIR, "frames")
CROP_INFO_FILE = os.path.join(BASE_DIR, "results", "crop_info.json")
TEMP_VGGT_DIR = os.path.join(DATA_DIR, "vggt_ready_frames")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

MAX_FRAMES = 20      # VGGT memory limit
VGGT_SIZE = 518      # Required input size

sys.path.insert(0, REPO_PATH)


# SETUP


def setup_environment():
    """Import VGGT modules."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        print("VGGT modules imported")
        return VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri
    except ImportError as e:
        print(f"ERROR: Could not import VGGT: {e}")
        print("Make sure VGGT is installed in the 'vggt' folder")
        sys.exit(1)

VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri = setup_environment()

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU - this will be slow!")

print("=" * 70)


# STEP 1: LOAD FRAMES FROM SCRIPT 1

def load_frames():
    """
    Load frames from Script 1's output directory.
    Select MAX_FRAMES evenly spaced throughout the video.
    """
    print(f"\n--- Step 1: Loading Frames ---")
    
    if not os.path.exists(FRAME_DIR):
        print(f"ERROR: Frames not found: {FRAME_DIR}")
        print("Run Script 1 first!")
        sys.exit(1)
    
    if not os.path.exists(CROP_INFO_FILE):
        print(f"ERROR: Crop info not found: {CROP_INFO_FILE}")
        print("Run Script 1 first!")
        sys.exit(1)
    
    # Find all frames
    all_frames = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
    
    if len(all_frames) == 0:
        print("ERROR: No frames found!")
        sys.exit(1)
    
    print(f"  Found {len(all_frames)} frames")
    
    # Select evenly spaced frames
    num_select = min(len(all_frames), MAX_FRAMES)
    indices = np.linspace(0, len(all_frames) - 1, num_select, dtype=int)
    
    selected_paths = [all_frames[i] for i in indices]
    
    # Extract Script 1 indices from filenames (format: 00000.jpg)
    script1_indices = []
    for path in selected_paths:
        filename = os.path.basename(path)
        idx = int(os.path.splitext(filename)[0])
        script1_indices.append(idx)
    
    print(f"✓ Selected {len(selected_paths)} frames for 3D reconstruction")
    print(f"  Script1 indices: {script1_indices[0]} to {script1_indices[-1]}")
    
    return selected_paths, script1_indices



# STEP 2: PREPROCESS FOR VGGT


def preprocess_frames(frame_paths):
    """
    Resize frames to 518x518 for VGGT.
    
    VGGT requires fixed 518x518 input. We:
    1. Scale image so largest dimension = 518
    2. Pad with black to make exactly 518x518
    """
    print(f"\n--- Step 2: Preprocessing ({VGGT_SIZE}x{VGGT_SIZE}) ---")
    
    # Clean up temp directory
    if os.path.exists(TEMP_VGGT_DIR):
        shutil.rmtree(TEMP_VGGT_DIR)
    os.makedirs(TEMP_VGGT_DIR, exist_ok=True)
    
    processed_paths = []
    
    for i, path in enumerate(tqdm(frame_paths, desc="  Resizing")):
        img = cv2.imread(path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Scale to fit in VGGT_SIZE
        scale = VGGT_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to VGGT_SIZE x VGGT_SIZE
        pad_w = (VGGT_SIZE - new_w) // 2
        pad_h = (VGGT_SIZE - new_h) // 2
        
        img = cv2.copyMakeBorder(
            img,
            top=pad_h,
            bottom=VGGT_SIZE - new_h - pad_h,
            left=pad_w,
            right=VGGT_SIZE - new_w - pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        out_path = os.path.join(TEMP_VGGT_DIR, f"{i:05d}.jpg")
        cv2.imwrite(out_path, img)
        processed_paths.append(out_path)
    
    print(f"✓ Preprocessed {len(processed_paths)} frames")
    return processed_paths



# STEP 3: RUN VGGT


def run_vggt(image_paths):
    """
    Run VGGT neural network to estimate depth and camera poses.
    """
    print(f"\n--- Step 3: Running VGGT Inference ---")
    
    # Load/download model
    checkpoint = os.path.join(CHECKPOINT_DIR, "vggt_model.pt")
    model = VGGT()
    
    if not os.path.exists(checkpoint):
        print("  Downloading VGGT weights (4GB)...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=DEVICE,
            progress=True
        )
        torch.save(state, checkpoint)
    
    print("  Loading model...", end="", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(" Done!")
    
    # Load images
    print(f"  Loading {len(image_paths)} images...", end="", flush=True)
    images = load_and_preprocess_images(image_paths).to(DEVICE)
    print(f" Shape: {images.shape}")
    
    # Run inference with mixed precision on GPU
    if DEVICE.type == "cuda":
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        ctx = nullcontext()
    
    print("  Running inference...", end="", flush=True)
    start = time.time()
    
    with torch.no_grad(), ctx:
        preds = model(images)
    
    print(f" Done! ({time.time() - start:.1f}s)")
    
    # Convert pose encoding to matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        preds["pose_enc"],
        images.shape[-2:]
    )
    
    # Convert to numpy
    def to_numpy(x):
        return x.float().cpu().numpy()
    
    results = {
        "images": to_numpy(images),
        "depth": to_numpy(preds["depth"]),
        "depth_conf": to_numpy(preds["depth_conf"]),
        "extrinsic": to_numpy(extrinsic),
        "intrinsic": to_numpy(intrinsic),
    }
    
    print(f"\n  Output shapes:")
    for name, arr in results.items():
        print(f"    {name}: {arr.shape}")
    
    return results



# STEP 4: PROCESS AND SAVE


def save_data(results, script1_indices):
    """
    Process VGGT output and save for Script 3.
    """
    print(f"\n--- Step 4: Saving Results ---")
    
    depth = results["depth"]
    extrinsic = results["extrinsic"]
    intrinsic = results["intrinsic"]
    images = results["images"]
    depth_conf = results["depth_conf"]
    
    print(f"  Original shapes:")
    print(f"    Depth: {depth.shape}")
    print(f"    Extrinsic: {extrinsic.shape}")
    print(f"    Intrinsic: {intrinsic.shape}")
    
    # Remove size-1 dimensions
    depth = np.squeeze(depth)
    extrinsic = np.squeeze(extrinsic)
    intrinsic = np.squeeze(intrinsic)
    images = np.squeeze(images)
    depth_conf = np.squeeze(depth_conf)
    
    print(f"  After squeeze:")
    print(f"    Depth: {depth.shape}")
    print(f"    Extrinsic: {extrinsic.shape}")
    print(f"    Intrinsic: {intrinsic.shape}")
    
    # Handle single frame edge case
    if depth.ndim == 2:
        depth = depth[np.newaxis, ...]
        extrinsic = extrinsic[np.newaxis, ...]
        intrinsic = intrinsic[np.newaxis, ...]
        images = images[np.newaxis, ...]
        depth_conf = depth_conf[np.newaxis, ...]
    
    if depth.ndim != 3:
        print(f"  ERROR: Unexpected depth shape: {depth.shape}")
        sys.exit(1)
    
    num_frames, H, W = depth.shape
    print(f"\n  Processing {num_frames} frames ({H}x{W})")
    
    # Convert 3x4 extrinsics to 4x4 (needed for matrix inversion)
    extr_rows, extr_cols = extrinsic.shape[-2:]
    is_3x4 = (extr_rows == 3 and extr_cols == 4)
    
    if is_3x4:
        print("  Converting extrinsics from 3x4 to 4x4...")
        extrinsics_4x4 = np.zeros((num_frames, 4, 4))
        for i in range(num_frames):
            extrinsics_4x4[i] = np.eye(4)
            extrinsics_4x4[i, :3, :] = extrinsic[i]
    else:
        extrinsics_4x4 = extrinsic
    
    # Create point cloud for visualization
    print("  Creating point cloud for visualization...")
    all_points = []
    all_colors = []
    all_conf = []
    
    for i in tqdm(range(num_frames), desc="  Unprojecting"):
        depth_i = depth[i]
        extr_i = extrinsics_4x4[i]
        intr_i = intrinsic[i]
        img_i = images[i]
        conf_i = depth_conf[i]
        
        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Camera intrinsics
        fx, fy = intr_i[0, 0], intr_i[1, 1]
        cx, cy = intr_i[0, 2], intr_i[1, 2]
        
        # Unproject to camera coordinates
        z = depth_i
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        
        cam_points = np.stack([x_cam, y_cam, z], axis=-1)
        
        # Transform to world coordinates
        cam_to_world = np.linalg.inv(extr_i)
        
        points_flat = cam_points.reshape(-1, 3)
        ones = np.ones((points_flat.shape[0], 1))
        points_homo = np.hstack([points_flat, ones])
        
        world_points = (cam_to_world @ points_homo.T).T[:, :3]
        
        # Colors (images are C,H,W format, values 0-1)
        colors = (img_i.transpose(1, 2, 0).reshape(-1, 3) * 255).astype(np.uint8)
        
        all_points.append(world_points)
        all_colors.append(colors)
        all_conf.append(conf_i.reshape(-1))
    
    # Combine all frames
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    conf = np.hstack(all_conf)
    
    print(f"\n  Total points: {len(points):,}")
    
    # Filter invalid points
    valid = np.isfinite(points).all(axis=1) & (conf > 0)
    points = points[valid]
    colors = colors[valid]
    conf = conf[valid]
    
    print(f"  Valid points: {len(points):,}")
    
    # Center point cloud
    center = np.median(points, axis=0)
    points = points - center
    
    # Save everything
    save_path = os.path.join(OUTPUT_DIR, "reconstruction.npz")
    
    np.savez_compressed(
        save_path,
        
        # Point cloud for visualization
        points=points,
        colors=colors,
        confidence=conf,
        center=center,
        
        # Data for Script 3 (electrode projection)
        images=images,
        depth=depth,
        depth_conf=depth_conf,
        extrinsics=extrinsics_4x4,
        intrinsics=intrinsic,
        
        # Frame mapping: VGGT index ↔ Script 1 index
        # IMPORTANT: Script 3 uses this to match tracking data to depth data
        frame_mapping_keys=np.arange(len(script1_indices)),     # VGGT indices [0,1,2,...]
        frame_mapping_values=np.array(script1_indices)          # Script1 indices [0,13,26,...]
    )
    
    print(f"\n Saved: {save_path}")
    
    return points, colors, conf


# STEP 5: OPTIONAL VIEWER


def launch_viewer(points, colors, conf):
    """Launch interactive 3D viewer using viser."""
    
    try:
        import viser
    except ImportError:
        print("\n Viser not installed. Skipping viewer.")
        print("  Install with: pip install viser")
        return
    
    port = 8080
    print(f"\n--- Viewer: http://localhost:{port} ---")
    
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    
    with server.gui.add_folder("Settings"):
        conf_slider = server.gui.add_slider("Confidence (%)", 0, 99, 1, 30)
        size_slider = server.gui.add_slider("Point Size", 0.001, 0.015, 0.001, 0.005)
    
    with server.gui.add_folder("Cropping"):
        cx = server.gui.add_slider("Clip X", 0.1, 3.0, 0.05, 2.0)
        cy = server.gui.add_slider("Clip Y", 0.1, 3.0, 0.05, 2.0)
        cz = server.gui.add_slider("Clip Z", 0.1, 3.0, 0.05, 2.0)
    
    pcd = server.scene.add_point_cloud("head", points, colors, size_slider.value)
    
    def update(_=None):
        mask = (
            (conf >= np.percentile(conf, conf_slider.value)) &
            (np.abs(points[:, 0]) < cx.value) &
            (np.abs(points[:, 1]) < cy.value) &
            (np.abs(points[:, 2]) < cz.value)
        )
        pcd.points = points[mask]
        pcd.colors = colors[mask]
        pcd.point_size = size_slider.value
    
    for slider in [conf_slider, size_slider, cx, cy, cz]:
        slider.on_update(update)
    update()
    
    print("Viewer ready! Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down viewer...")



# MAIN


def main():
    total_start = time.time()
    
    # Run pipeline
    frame_paths, script1_indices = load_frames()
    processed_paths = preprocess_frames(frame_paths)
    results = run_vggt(processed_paths)
    points, colors, conf = save_data(results, script1_indices)
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("SCRIPT 2 COMPLETE!")
    print("=" * 70)
    print(f"\nTime: {total_time / 60:.1f} minutes")
    print(f"\nOutput: {os.path.join(OUTPUT_DIR, 'reconstruction.npz')}")
    print(f"\nNext: Run Script 3 (2D→3D projection)")
    
    # Optional viewer
    launch_viewer(points, colors, conf)


if __name__ == "__main__":
    main()