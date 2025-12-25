"""
VGGT 3D Head Reconstruction 

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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = os.path.join(BASE_DIR, "vggt")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "vggt_output")

SCRIPT1_FRAMES_DIR = os.path.join(BASE_DIR, "frames")
CROP_INFO_FILE = os.path.join(BASE_DIR, "results", "crop_info.json")
TEMP_VGGT_DIR = os.path.join(DATA_DIR, "vggt_ready_frames")

MAX_FRAMES_FOR_3D = 20
TARGET_SIZE = 518

sys.path.insert(0, REPO_PATH)



# SETUP


def setup_environment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        print(" VGGT modules imported")
        return VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import VGGT.\n   {e}")
        sys.exit(1)


VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri = setup_environment()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print(" Using CPU. This will be slow!")


# 1. LOAD DATA FROM SCRIPT 1


def load_script1_data():
    print(f"\n--- 1. Loading Data from Script 1 ---")
    
    if not os.path.exists(SCRIPT1_FRAMES_DIR):
        print(f"  Frames folder not found: {SCRIPT1_FRAMES_DIR}")
        print("     Run Script 1 first!")
        sys.exit(1)

    if not os.path.exists(CROP_INFO_FILE):
        print(f"  Crop info not found: {CROP_INFO_FILE}")
        print("     Run Script 1 first!")
        sys.exit(1)
    
    all_frames = sorted(glob.glob(os.path.join(SCRIPT1_FRAMES_DIR, "*.jpg")))
    if len(all_frames) == 0:
        print(" Frames folder is empty.")
        sys.exit(1)
    
    print(f"  Found {len(all_frames)} frames")
    
    num_select = min(len(all_frames), MAX_FRAMES_FOR_3D)
    indices = np.linspace(0, len(all_frames) - 1, num_select, dtype=int)
    
    selected_paths = [all_frames[i] for i in indices]
    
    script1_indices = []
    for p in selected_paths:
        fname = os.path.basename(p)
        idx = int(os.path.splitext(fname)[0])
        script1_indices.append(idx)
    
    print(f" Selected {len(selected_paths)} frames for 3D reconstruction")
    
    return selected_paths, script1_indices



# 2. PREPROCESS FOR VGGT


def preprocess_frames(frame_paths):
    print(f"\n--- 2. Preprocessing for VGGT ({TARGET_SIZE}x{TARGET_SIZE}) ---")
    
    if os.path.exists(TEMP_VGGT_DIR):
        shutil.rmtree(TEMP_VGGT_DIR)
    os.makedirs(TEMP_VGGT_DIR, exist_ok=True)
    
    processed_paths = []
    
    for i, path in enumerate(tqdm(frame_paths, desc="  Resizing")):
        img = cv2.imread(path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        scale = TARGET_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pad_w = (TARGET_SIZE - new_w) // 2
        pad_h = (TARGET_SIZE - new_h) // 2
        img = cv2.copyMakeBorder(
            img,
            pad_h, TARGET_SIZE - new_h - pad_h,
            pad_w, TARGET_SIZE - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        out_path = os.path.join(TEMP_VGGT_DIR, f"{i:05d}.jpg")
        cv2.imwrite(out_path, img)
        processed_paths.append(out_path)
    
    print(f"  Preprocessed {len(processed_paths)} frames")
    return processed_paths



# 3. RUN VGGT MODEL


def run_vggt(image_paths):
    print(f"\n--- 3. Running VGGT Inference ---")
    
    checkpoint = os.path.join(BASE_DIR, "checkpoints", "vggt_model.pt")
    model = VGGT()
    
    if not os.path.exists(checkpoint):
        print("  ⬇ Downloading model weights (4GB)...")
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        
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
    
    print(f"  Loading {len(image_paths)} images...", end="", flush=True)
    images = load_and_preprocess_images(image_paths).to(DEVICE)
    print(f" Shape: {images.shape}")
    
    if DEVICE.type == "cuda":
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        ctx = nullcontext()
    
    print("  Running inference...", end="", flush=True)
    start = time.time()
    with torch.no_grad(), ctx:
        preds = model(images)
    print(f" Done ({time.time() - start:.1f}s)")
    
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        preds["pose_enc"], images.shape[-2:]
    )
    
    def to_numpy(x):
        return x.float().cpu().numpy()
    
    results = {
        "images": to_numpy(images),
        "depth": to_numpy(preds["depth"]),
        "depth_conf": to_numpy(preds["depth_conf"]),
        "extrinsic": to_numpy(extrinsic),
        "intrinsic": to_numpy(intrinsic),
    }
    
    print(f"\n  Raw tensor shapes:")
    for k, v in results.items():
        print(f"    {k}: {v.shape}")
    
    return results



# 4. SAVE RESULTS


def save_data(results, script1_indices):
    print(f"\n--- 4. Saving Results ---")
    
    depth = results["depth"]
    extrinsic = results["extrinsic"]
    intrinsic = results["intrinsic"]
    images = results["images"]
    depth_conf = results["depth_conf"]
    
    print(f"  Original shapes:")
    print(f"    Depth: {depth.shape}")
    print(f"    Extrinsic: {extrinsic.shape}")
    print(f"    Intrinsic: {intrinsic.shape}")
    
    # Squeeze all size-1 dimensions
    depth = np.squeeze(depth)
    extrinsic = np.squeeze(extrinsic)
    intrinsic = np.squeeze(intrinsic)
    images = np.squeeze(images)
    depth_conf = np.squeeze(depth_conf)
    
    print(f"  After squeeze:")
    print(f"    Depth: {depth.shape}")
    print(f"    Extrinsic: {extrinsic.shape}")
    print(f"    Intrinsic: {intrinsic.shape}")
    
    # Handle single frame
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
    print(f"\n  Processing {num_frames} frames of size {H}x{W}")
    
    # Check extrinsic format (3x4 or 4x4)
    extr_rows, extr_cols = extrinsic.shape[-2:]
    is_3x4 = (extr_rows == 3 and extr_cols == 4)
    print(f"  Extrinsic format: {extr_rows}x{extr_cols}" + (" → will convert to 4x4" if is_3x4 else ""))
    
    # Unproject each frame
    all_points = []
    all_colors = []
    all_conf = []
    extrinsics_4x4 = np.zeros((num_frames, 4, 4))
    
    for i in tqdm(range(num_frames), desc="  Unprojecting"):
        depth_i = depth[i]
        extr_i = extrinsic[i]
        intr_i = intrinsic[i]
        img_i = images[i]
        conf_i = depth_conf[i]
        
        # Convert 3x4 to 4x4 if needed
        if is_3x4:
            extr_4x4 = np.eye(4)
            extr_4x4[:3, :] = extr_i
        else:
            extr_4x4 = extr_i
        
        extrinsics_4x4[i] = extr_4x4
        
        # Create pixel grid
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
        cam_to_world = np.linalg.inv(extr_4x4)
        
        points_flat = cam_points.reshape(-1, 3)
        ones = np.ones((points_flat.shape[0], 1))
        points_homo = np.hstack([points_flat, ones])
        
        world_points = (cam_to_world @ points_homo.T).T[:, :3]
        
        # Colors
        colors_i = (img_i.transpose(1, 2, 0).reshape(-1, 3) * 255).astype(np.uint8)
        
        # Confidence
        conf_flat = conf_i.reshape(-1)
        
        all_points.append(world_points)
        all_colors.append(colors_i)
        all_conf.append(conf_flat)
    
    # Combine
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    conf = np.hstack(all_conf)
    
    print(f"\n  Total points: {len(points):,}")
    
    # Filter invalid
    valid = np.isfinite(points).all(axis=1) & (conf > 0)
    points = points[valid]
    colors = colors[valid]
    conf = conf[valid]
    
    print(f"  Valid points: {len(points):,}")
    
    # Center
    center = np.median(points, axis=0)
    points = points - center
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, "reconstruction.npz")
    np.savez_compressed(
        save_path,
        # Visualization
        points=points,
        colors=colors,
        confidence=conf,
        center=center,
        
        # Math data (squeezed, 4x4 extrinsics)
        images=images,
        depth=depth,
        depth_conf=depth_conf,
        extrinsics=extrinsics_4x4,
        intrinsics=intrinsic,
        
        # Frame mapping
        frame_mapping_keys=np.arange(len(script1_indices)),
        frame_mapping_values=np.array(script1_indices)
    )
    
    print(f"\n  Saved: {save_path}")
    return points, colors, conf



# 5. VIEWER


def launch_viewer(points, colors, conf):
    try:
        import viser
    except ImportError:
        print("\n⚠ Viser not installed. Skipping viewer.")
        print("  Install with: pip install viser")
        return

    port = 8080
    print(f"\n--- Starting Viewer on http://localhost:{port} ---")
    
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
    
    for s in [conf_slider, size_slider, cx, cy, cz]:
        s.on_update(update)
    update()
    
    print("  ✓ Viewer ready! Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down...")



# MAIN


def main():
    print("=" * 70)
    print("VGGT 3D RECONSTRUCTION (PIPELINE STEP 2)")
    print("=" * 70)
    
    total_start = time.time()
    
    paths, indices = load_script1_data()
    ready_paths = preprocess_frames(paths)
    results = run_vggt(ready_paths)
    pts, cols, cnf = save_data(results, indices)
    
    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE! Total time: {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}")
    
    launch_viewer(pts, cols, cnf)


if __name__ == "__main__":
    main()