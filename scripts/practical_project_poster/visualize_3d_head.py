"""
3D Head + Electrode Visualization for Poster
=============================================
Creates a publication-quality 3D rendering showing:
- VGGT reconstructed head point cloud (colored from video frames)
- Electrode positions as labeled spheres
- Landmark positions (NAS, LPA, RPA) as distinct markers

Two views generated: A) Side/front view  B) Top-down view

Requirements:
    pip install pyvista numpy

Usage:
    python visualize_3d_head.py

    It will prompt you to select which video result to visualize.
"""

import os
import sys
import json
import numpy as np
import pickle
from glob import glob

# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_BASE_DIR = "results"
DATA_DIR = "data"              # Where vggt_ready_frames_<VIDEO> folders are

# Frame folder naming pattern: data/vggt_ready_frames_<VIDEO_NAME>/
# e.g. data/vggt_ready_frames_IMG_3841/

# Electrode sphere size (mm) — adjust for your head scale
ELECTRODE_SPHERE_RADIUS = 3.0
LANDMARK_SPHERE_RADIUS = 4.0

# Colors
COLOR_ELECTRODE = '#DA91BA'     # UOL Pink (Pantone 204)
COLOR_NAS = '#EE7100'           # Orange
COLOR_LPA = '#00ABD9'           # Blue
COLOR_RPA = '#00ABD9'           # Blue
COLOR_INION = '#EE7100'         # Orange
COLOR_POINT_CLOUD = None        # None = use real RGB from video; or set e.g. '#E8D5C4' for skin tone

# Output
OUTPUT_DPI = 600
SAVE_PDF = True


# ============================================================
# DATA LOADING
# ============================================================

def select_video():
    """Let user pick which video result to visualize."""
    video_dirs = []
    for item in sorted(os.listdir(RESULTS_BASE_DIR)):
        item_path = os.path.join(RESULTS_BASE_DIR, item)
        json_path = os.path.join(item_path, "electrodes_3d.json")
        recon_path = os.path.join(item_path, "vggt_output", "reconstruction.npz")
        if os.path.isdir(item_path) and os.path.exists(json_path):
            has_recon = os.path.exists(recon_path)
            video_dirs.append((item, has_recon))

    if not video_dirs:
        print("No video results found in results/")
        sys.exit(1)

    print("\nAvailable video results:")
    for i, (name, has_recon) in enumerate(video_dirs, 1):
        recon_status = "[has point cloud]" if has_recon else "[electrodes only]"
        print(f"  [{i}] {name} {recon_status}")

    while True:
        try:
            idx = int(input(f"\nSelect (1-{len(video_dirs)}): ")) - 1
            if 0 <= idx < len(video_dirs):
                break
        except ValueError:
            pass

    selected = video_dirs[idx][0]
    return os.path.join(RESULTS_BASE_DIR, selected)


def load_electrodes(results_dir):
    """Load electrode and landmark positions from electrodes_3d.json."""
    json_path = os.path.join(results_dir, "electrodes_3d.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    electrodes = {}
    landmarks = {}

    for name, info in data.get("landmarks", {}).items():
        landmarks[name.upper()] = np.array(info["position"])

    for name, info in data.get("electrodes", {}).items():
        electrodes[name.upper()] = np.array(info["position"])

    print(f"  Loaded {len(electrodes)} electrodes, {len(landmarks)} landmarks")
    return electrodes, landmarks


def load_point_cloud(results_dir):
    """Build 3D point cloud by unprojecting VGGT depth maps.
    
    reconstruction.npz contains: depth, extrinsics, intrinsics, frame_mapping.
    We unproject each pixel using depth + camera intrinsics, then transform
    to world coordinates using extrinsics. Colors come from the original frames.
    """
    recon_path = os.path.join(results_dir, "vggt_output", "reconstruction.npz")
    if not os.path.exists(recon_path):
        print("  No reconstruction.npz found — will show electrodes only")
        return None, None

    recon = np.load(recon_path)
    depths = np.squeeze(recon["depth"])          # (N_frames, H, W)
    intrinsics = np.squeeze(recon["intrinsics"])  # (N_frames, 3, 3)
    extrinsics = np.squeeze(recon["extrinsics"])  # (N_frames, 4, 4) or (N, 3, 4)
    
    # Ensure 4x4 extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        E4 = np.zeros((extrinsics.shape[0], 4, 4))
        for i in range(extrinsics.shape[0]):
            E4[i] = np.eye(4)
            E4[i, :3, :] = extrinsics[i]
        extrinsics = E4

    n_frames = depths.shape[0]
    h, w = depths.shape[1], depths.shape[2]
    print(f"  Depth maps: {n_frames} frames, {h}x{w}")

    # Try to load original frames for coloring
    # Frames are stored in: data/vggt_ready_frames_<VIDEO_NAME>/
    video_name = os.path.basename(results_dir)
    frames_dir = os.path.join(DATA_DIR, f"vggt_ready_frames_{video_name}")
    
    # Fallback: check results dir too
    if not os.path.exists(frames_dir):
        frames_dir = os.path.join(results_dir, "frames")
    
    crop_path = os.path.join(results_dir, "crop_info.json")
    frame_images = None
    
    if os.path.exists(frames_dir):
        import cv2
        print(f"  Frames found: {frames_dir}")
        
        # Load crop info if available (may not exist for full-frame input)
        crop = {}
        if os.path.exists(crop_path):
            with open(crop_path, 'r') as f:
                crop = json.load(f)
        
        # Get frame mapping
        s1_indices = recon["frame_mapping_values"]
        
        # Load frames for color
        frame_files = sorted(glob(os.path.join(frames_dir, "*.png")) + 
                           glob(os.path.join(frames_dir, "*.jpg")))
        if frame_files:
            frame_images = {}
            for vggt_idx, s1_idx in enumerate(s1_indices):
                s1_idx = int(s1_idx)
                if s1_idx < len(frame_files):
                    img = cv2.imread(frame_files[s1_idx])
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Apply crop if crop_info exists
                        if crop:
                            cx, cy = crop.get("x", 0), crop.get("y", 0)
                            cw, ch = crop.get("w", img.shape[1]), crop.get("h", img.shape[0])
                            img = img[cy:cy+ch, cx:cx+cw]
                        # Resize to depth map resolution
                        img_resized = cv2.resize(img, (w, h))
                        frame_images[vggt_idx] = img_resized
            print(f"  Loaded {len(frame_images)} frames for coloring")
    else:
        print(f"  No frames found (looked in: {frames_dir})")
        print(f"  Point cloud will use skin-tone color instead of video RGB")

    # Unproject depth maps to 3D
    # Use a subset of frames and subsample spatially
    target_points = 200000
    frame_skip = max(1, n_frames // 8)
    pixel_skip = max(1, int(np.sqrt(n_frames * h * w / (target_points * max(1, n_frames // frame_skip)))))
    
    all_points = []
    all_colors = []
    
    # Create pixel grid (subsampled)
    vs, us = np.mgrid[0:h:pixel_skip, 0:w:pixel_skip]
    us_flat = us.flatten().astype(np.float64)
    vs_flat = vs.flatten().astype(np.float64)
    
    for fi in range(0, n_frames, frame_skip):
        K = intrinsics[fi]  # (3, 3)
        E = extrinsics[fi]  # (4, 4)
        
        # Get depth values at subsampled pixels
        d = depths[fi, ::pixel_skip, ::pixel_skip].flatten()
        
        # Filter valid depths
        valid = (d > 0) & np.isfinite(d)
        if valid.sum() == 0:
            continue
            
        d_valid = d[valid]
        u_valid = us_flat[valid]
        v_valid = vs_flat[valid]
        
        # Unproject: pixel + depth -> camera coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x_cam = (u_valid - cx) * d_valid / fx
        y_cam = (v_valid - cy) * d_valid / fy
        z_cam = d_valid
        
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)  # (N, 4)
        
        # Camera to world: P_world = E^{-1} @ P_cam
        E_inv = np.linalg.inv(E)
        pts_world = (E_inv @ pts_cam.T).T[:, :3]  # (N, 3)
        
        all_points.append(pts_world)
        
        # Get colors — must match points 1:1
        if frame_images is not None and fi in frame_images:
            img = frame_images[fi]
            u_int = np.clip(us_flat[valid].astype(int), 0, w - 1)
            v_int = np.clip(vs_flat[valid].astype(int), 0, h - 1)
            frame_colors = img[v_int, u_int]  # (N, 3)
            all_colors.append(frame_colors)
        elif frame_images is not None:
            # No image for this frame — fill with skin-tone so arrays stay aligned
            skin = np.array([232, 213, 196], dtype=np.uint8)  # #E8D5C4
            all_colors.append(np.tile(skin, (len(pts_world), 1)))
    
    if not all_points:
        print("  Could not generate point cloud from depth maps")
        return None, None
        
    points = np.concatenate(all_points, axis=0)
    rgb = np.concatenate(all_colors, axis=0) if all_colors else None
    
    # Remove statistical outliers
    centroid = np.median(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    threshold = np.percentile(dists, 95)
    mask = dists < threshold
    points = points[mask]
    if rgb is not None:
        rgb = rgb[mask]
    
    print(f"  Point cloud: {len(points)} points")
    return points, rgb


# ============================================================
# VISUALIZATION WITH PYVISTA
# ============================================================

def render_with_pyvista(points, rgb, electrodes, landmarks, results_dir):
    """High-quality 3D rendering using PyVista."""
    import pyvista as pv

    video_name = os.path.basename(results_dir)
    
    # Create plotter
    for view_name, camera_position, figsize in [
        ("side", "xz", (1600, 1200)),
        ("top", "xy", (1600, 1200)),
        ("front", "yz", (1600, 1200)),
    ]:
        pl = pv.Plotter(off_screen=True, window_size=figsize)
        pl.set_background('white')

        # Add point cloud OR a generic fallback head
        # FORCE A GENERIC HEAD
        # Combine electrodes and landmarks to find the absolute boundaries
        all_pts = np.array(list(electrodes.values()) + list(landmarks.values()))
        
        min_p = all_pts.min(axis=0)
        max_p = all_pts.max(axis=0)
        
        # Calculate the X (width) and Y (length) radii
        rx = (max_p[0] - min_p[0]) / 2.0
        ry = (max_p[1] - min_p[1]) / 2.0
        
        # Because electrodes only cover the top half of the head, we estimate 
        # the true Z (height) radius using the X and Y radii to make it proportional.
        rz = (rx + ry) / 2.0 
        
        # Find the true center
        cx = (max_p[0] + min_p[0]) / 2.0
        cy = (max_p[1] + min_p[1]) / 2.0
        
        # The highest electrode sits at the top of the head. 
        # So the center Z is the max Z minus the radius.
        cz = max_p[2] - rz
        
        # Scale to 94% so the electrodes pop out nicely on the surface
        scale = 0.94
        rx *= scale
        ry *= scale
        rz *= scale
        
        # Build and stretch the head
        head = pv.Sphere(radius=1.0, theta_resolution=100, phi_resolution=100)
        head.points[:, 0] *= rx  # Width
        head.points[:, 1] *= ry  # Length
        head.points[:, 2] *= rz  # Height
        head.points += np.array([cx, cy, cz]) # Move to calculated center
        
        color = COLOR_POINT_CLOUD or '#E8D5C4'
        pl.add_mesh(head, color=color, opacity=0.3, smooth_shading=True)

        # Add electrodes as spheres (no labels — landmarks only)
        for name, pos in electrodes.items():
            sphere = pv.Sphere(radius=ELECTRODE_SPHERE_RADIUS, center=pos)
            pl.add_mesh(sphere, color=COLOR_ELECTRODE, smooth_shading=True)

        # Add landmarks as larger, distinct spheres with prominent labels
        landmark_colors = {
            'NAS': COLOR_NAS, 'LPA': COLOR_LPA, 
            'RPA': COLOR_RPA, 'INION': COLOR_INION
        }
        for name, pos in landmarks.items():
            color = landmark_colors.get(name, '#FFD700')
            sphere = pv.Sphere(radius=LANDMARK_SPHERE_RADIUS, center=pos)
            pl.add_mesh(sphere, color=color, smooth_shading=True)
            # Label with white background for readability
            label_pos = pos + np.array([0, 0, LANDMARK_SPHERE_RADIUS * 2.5])
            pl.add_point_labels(
                pv.PolyData(label_pos.reshape(1, 3)),
                [name], font_size=18, text_color='black',
                shape_color='white', shape_opacity=0.7,
                bold=True, always_visible=True
            )

        # Camera
        pl.camera_position = camera_position
        pl.camera.zoom(1.2)

        # Save
        out_path = os.path.join(results_dir, f"3d_visualization_{view_name}.png")
        pl.screenshot(out_path, scale=2)  # 2x for high-res
        pl.close()
        print(f"  Saved: {out_path}")

    # Create combined A/B panel figure
    _create_panel_figure(results_dir, video_name)


def _create_panel_figure(results_dir, video_name):
    """Combine side + top views into a single A/B panel figure."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    side_path = os.path.join(results_dir, "3d_visualization_side.png")
    top_path = os.path.join(results_dir, "3d_visualization_top.png")

    if not os.path.exists(side_path) or not os.path.exists(top_path):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    img_side = mpimg.imread(side_path)
    img_top = mpimg.imread(top_path)

    ax1.imshow(img_side)
    ax1.set_title("A", fontsize=24, fontweight='bold', loc='left')
    ax1.axis('off')

    ax2.imshow(img_top)
    ax2.set_title("B", fontsize=24, fontweight='bold', loc='left')
    ax2.axis('off')

    plt.tight_layout()
    panel_path = os.path.join(results_dir, "3d_visualization_panel.png")
    plt.savefig(panel_path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    if SAVE_PDF:
        plt.savefig(panel_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved panel: {panel_path}")


# ============================================================
# FALLBACK: MATPLOTLIB 3D (if PyVista not available)
# ============================================================

def render_with_matplotlib(points, rgb, electrodes, landmarks, results_dir):
    """Fallback 3D rendering using matplotlib (lower quality but no extra deps)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    video_name = os.path.basename(results_dir)

    fig = plt.figure(figsize=(20, 10))

    for panel_idx, (elev, azim, title) in enumerate([
        (20, -60, "A"),    # Side view
        (90, 0, "B"),      # Top view
    ], 1):
        ax = fig.add_subplot(1, 2, panel_idx, projection='3d')

        # Point cloud
        if points is not None and len(points) > 0:
            # Further subsample for matplotlib (it's slow with many points)
            n = min(50000, len(points))
            idx = np.random.choice(len(points), n, replace=False)
            pts_sub = points[idx]

            if rgb is not None:
                colors_sub = rgb[idx] / 255.0
                ax.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2],
                          c=colors_sub, s=0.5, alpha=0.3)
            else:
                ax.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2],
                          c='#E8D5C4', s=0.5, alpha=0.3)

        # Electrodes (spheres only, no labels)
        for name, pos in electrodes.items():
            ax.scatter(*pos, c=COLOR_ELECTRODE, s=80, edgecolors='black',
                      linewidths=0.5, zorder=10, depthshade=False)

        # Landmarks — larger markers with prominent labels
        landmark_colors = {'NAS': COLOR_NAS, 'LPA': COLOR_LPA,
                          'RPA': COLOR_RPA, 'INION': COLOR_INION}
        for name, pos in landmarks.items():
            color = landmark_colors.get(name, '#FFD700')
            ax.scatter(*pos, c=color, s=150, edgecolors='black',
                      linewidths=1.5, zorder=11, marker='D', depthshade=False)
            ax.text(pos[0], pos[1], pos[2] + LANDMARK_SPHERE_RADIUS * 3,
                   name, fontsize=11, ha='center', fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor=color, alpha=0.8, linewidth=1.5))

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title, fontsize=20, fontweight='bold', loc='left')

        # Equal aspect ratio
        all_pts = np.array(list(electrodes.values()) + list(landmarks.values()))
        mid = all_pts.mean(axis=0)
        span = max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2 * 1.2
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)

    plt.suptitle(f"3D Reconstruction — {video_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(results_dir, "3d_visualization_panel.png")
    plt.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    if SAVE_PDF:
        plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("3D HEAD + ELECTRODE VISUALIZATION")
    print("=" * 50)

    # Select video
    results_dir = select_video()
    video_name = os.path.basename(results_dir)
    print(f"\nProcessing: {video_name}")

    # Load data
    print("\n--- Loading data ---")
    electrodes, landmarks = load_electrodes(results_dir)
    points, rgb = load_point_cloud(results_dir)

    # Render
    print("\n--- Rendering ---")
    try:
        import pyvista
        print("  Using PyVista (high quality)")
        render_with_pyvista(points, rgb, electrodes, landmarks, results_dir)
    except ImportError:
        print("  PyVista not found, using matplotlib fallback")
        print("  For better quality: pip install pyvista")
        render_with_matplotlib(points, rgb, electrodes, landmarks, results_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()