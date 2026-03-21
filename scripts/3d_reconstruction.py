"""
VGGT 3D Reconstruction — Turntable Video Generator
====================================================

Creates a rotating video of the VGGT point cloud reconstruction,
suitable for embedding on a website or in presentations.

Usage:
    python vggt_turntable_video.py

Edit VIDEO_NAME below to match your results folder name.

Requirements:
    pip install open3d opencv-python numpy
"""

import os
import sys
import glob
import numpy as np
import cv2

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# Paths — edit these to match your setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

VIDEO_NAME = "IMG_3841"  # ← Change to your video folder name

RESULTS_DIR = os.path.join(BASE_DIR, "results", VIDEO_NAME)
RECON_FILE = os.path.join(RESULTS_DIR, "vggt_output", "reconstruction.npz")
FRAME_DIR = os.path.join(BASE_DIR, "frames")

# Video output
OUTPUT_VIDEO = os.path.join(RESULTS_DIR, f"{VIDEO_NAME}_reconstruction_turntable.mp4")

# Rendering settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30
DURATION_SECONDS = 8          # Total video length
N_ROTATIONS = 1               # Number of full 360° rotations
POINT_SIZE = 3.0              # Point radius in pixels (2-3 looks good at 1080p)
BG_COLOR = [0.05, 0.05, 0.08] # Near-black background (matches Viser dark theme)
MAX_DISTANCE = 1.5            # Discard points farther than this from centroid
ELEVATION_DEG = 20            # Camera elevation above horizon (degrees)
CAMERA_DISTANCE_FACTOR = 2.0  # Multiplied by point cloud radius for camera distance

# Frame subsampling for point cloud density (1 = all pixels, 2 = every other, etc.)
PIXEL_STEP = 2

VGGT_SIZE = 518  # Must match the value used in Script1


# ── LOAD RECONSTRUCTION ───────────────────────────────────────────────────────

def load_reconstruction():
    """Load depth, extrinsics, intrinsics from reconstruction.npz."""
    if not os.path.exists(RECON_FILE):
        print(f"ERROR: reconstruction.npz not found at:\n  {RECON_FILE}")
        print("Make sure VIDEO_NAME is correct and Script1 has been run.")
        sys.exit(1)

    data = np.load(RECON_FILE)
    depth = data["depth"]          # (N, H, W)
    extrinsics = data["extrinsics"]  # (N, 4, 4)
    intrinsics = data["intrinsics"]  # (N, 3, 3)
    s1_indices = data["frame_mapping_values"]  # original frame indices

    print(f"Loaded reconstruction: {depth.shape[0]} frames, depth {depth.shape[1]}×{depth.shape[2]}")
    return depth, extrinsics, intrinsics, s1_indices


# ── LOAD FRAME COLORS ─────────────────────────────────────────────────────────

def load_frame_colors(s1_indices):
    """Load original frames to extract RGB colors for the point cloud."""
    if not os.path.exists(FRAME_DIR):
        print(f"  Frames directory not found: {FRAME_DIR}")
        return None

    colors_per_frame = []
    for s1_idx in s1_indices:
        path = os.path.join(FRAME_DIR, f"{s1_idx:05d}.jpg")
        if not os.path.exists(path):
            print(f"  Frame not found: {path}")
            return None
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize + pad to VGGT_SIZE (same as Script1)
        h, w = img.shape[:2]
        scale = VGGT_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        pad_h = (VGGT_SIZE - img.shape[0]) // 2
        pad_w = (VGGT_SIZE - img.shape[1]) // 2
        img = cv2.copyMakeBorder(
            img, pad_h, VGGT_SIZE - img.shape[0] - pad_h,
            pad_w, VGGT_SIZE - img.shape[1] - pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        colors_per_frame.append(img)

    print(f"  Loaded {len(colors_per_frame)} frames for RGB coloring")
    return colors_per_frame


# ── BUILD POINT CLOUD ─────────────────────────────────────────────────────────

def build_point_cloud(depth, extrinsics, intrinsics, colors_per_frame=None):
    """
    Unproject depth maps to 3D world coordinates.

    Returns:
        points: (M, 3) float64 array
        colors: (M, 3) float64 array in [0, 1]
    """
    all_points = []
    all_colors = []

    n_frames = depth.shape[0]
    print(f"\nBuilding point cloud from {n_frames} frames (step={PIXEL_STEP})...")

    for i in range(n_frames):
        d = depth[i]          # (H, W)
        K = intrinsics[i]     # (3, 3)
        E = extrinsics[i]     # (3, 4) or (4, 4)

        # Convert 3×4 → 4×4 if needed (reconstruction.npz stores raw 3×4)
        if E.shape == (3, 4):
            E4 = np.eye(4)
            E4[:3, :] = E
            E = E4

        E_inv = np.linalg.inv(E)

        H, W = d.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create pixel grid (subsampled)
        vs, us = np.mgrid[0:H:PIXEL_STEP, 0:W:PIXEL_STEP]
        vs = vs.flatten()
        us = us.flatten()
        zs = d[vs, us]

        # Filter invalid depths
        valid = (zs > 0) & np.isfinite(zs)
        vs, us, zs = vs[valid], us[valid], zs[valid]

        # Unproject to camera space
        x_cam = (us - cx) * zs / fx
        y_cam = (vs - cy) * zs / fy
        ones = np.ones_like(zs)
        P_cam = np.stack([x_cam, y_cam, zs, ones], axis=1)  # (M, 4)

        # Transform to world space
        P_world = (E_inv @ P_cam.T).T[:, :3]  # (M, 3)
        all_points.append(P_world)

        # Colors
        if colors_per_frame is not None:
            frame_rgb = colors_per_frame[i]
            cs = frame_rgb[vs, us].astype(np.float64) / 255.0
        else:
            # Depth-based coloring (blue = near, red = far)
            z_norm = (zs - zs.min()) / (zs.max() - zs.min() + 1e-8)
            cs = np.stack([z_norm, 0.3 * np.ones_like(z_norm), 1.0 - z_norm], axis=1)
        all_colors.append(cs)

        print(f"  Frame {i+1}/{n_frames}: {len(P_world):,} points", end="\r")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"\n  Total: {len(points):,} points before filtering")

    # Filter by distance from centroid
    centroid = np.median(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    mask = dists < MAX_DISTANCE
    points = points[mask]
    colors = colors[mask]
    print(f"  After distance filter (<{MAX_DISTANCE}): {len(points):,} points")

    return points, colors


# ── RENDER TURNTABLE VIDEO ────────────────────────────────────────────────────

def render_turntable(points, colors):
    """
    Render a turntable video using pure numpy projection + OpenCV drawing.

    No Open3D / GPU / display needed — works on any machine.
    The object rotates in place while the camera stays fixed.
    Points are z-sorted (painter's algorithm) so closer points draw on top.
    """
    n_frames = FPS * DURATION_SECONDS
    total_angle = 360.0 * N_ROTATIONS

    # Center the point cloud at origin
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid

    # Compute bounding radius for camera placement
    radius = np.percentile(np.linalg.norm(pts_centered, axis=1), 95)
    cam_dist = radius * CAMERA_DISTANCE_FACTOR

    # Virtual camera intrinsics
    focal = VIDEO_WIDTH * 1.2  # Controls field of view
    cx, cy = VIDEO_WIDTH / 2.0, VIDEO_HEIGHT / 2.0

    # Camera extrinsic: positioned behind and above, looking at origin
    elev_rad = np.deg2rad(ELEVATION_DEG)

    # Build look-at view matrix
    eye = np.array([
        0.0,
        -cam_dist * np.sin(elev_rad),
        cam_dist * np.cos(elev_rad),
    ])
    forward = -eye / np.linalg.norm(eye)        # toward origin
    right = np.cross(forward, np.array([0, -1, 0]))
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)

    # View matrix (world → camera)
    R_view = np.stack([right, -up, forward], axis=0)  # (3, 3)
    t_view = -R_view @ eye                              # (3,)

    # Background color as uint8
    bg = np.array(BG_COLOR, dtype=np.float64)
    bg_uint8 = (np.clip(bg, 0, 1) * 255).astype(np.uint8)

    # Precompute colors as uint8 BGR for OpenCV
    colors_bgr = (np.clip(colors[:, ::-1], 0, 1) * 255).astype(np.uint8)

    # Point radius in pixels
    pt_radius = max(1, int(POINT_SIZE))

    print(f"\nRendering turntable video (numpy + OpenCV):")
    print(f"  Resolution: {VIDEO_WIDTH}×{VIDEO_HEIGHT}")
    print(f"  Duration:   {DURATION_SECONDS}s @ {FPS} fps = {n_frames} frames")
    print(f"  Rotations:  {N_ROTATIONS} × 360°")
    print(f"  Points:     {len(points):,}")
    print(f"  Output:     {OUTPUT_VIDEO}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    if not writer.isOpened():
        print("ERROR: Could not open video writer.")
        sys.exit(1)

    for frame_idx in range(n_frames):
        angle_deg = (frame_idx / n_frames) * total_angle
        angle_rad = np.deg2rad(angle_deg)

        # Rotation matrix around Y axis (vertical)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R_turntable = np.array([
            [ cos_a, 0, sin_a],
            [ 0,     1, 0    ],
            [-sin_a, 0, cos_a],
        ])

        # Rotate points, then apply view transform
        pts_rot = (R_turntable @ pts_centered.T).T       # (N, 3)
        pts_cam = (R_view @ pts_rot.T).T + t_view         # (N, 3)

        # Keep only points in front of camera
        z = pts_cam[:, 2]
        in_front = z > 0.01
        pts_cam = pts_cam[in_front]
        z = z[in_front]
        cols = colors_bgr[in_front]

        # Project to 2D
        u = (focal * pts_cam[:, 0] / z + cx).astype(np.int32)
        v = (focal * pts_cam[:, 1] / z + cy).astype(np.int32)

        # Clip to image bounds
        visible = (u >= 0) & (u < VIDEO_WIDTH) & (v >= 0) & (v < VIDEO_HEIGHT)
        u, v, z, cols = u[visible], v[visible], z[visible], cols[visible]

        # Sort by depth — farthest first (painter's algorithm)
        order = np.argsort(-z)
        u, v, cols = u[order], v[order], cols[order]

        # Draw — fast numpy splatting (no per-point loop)
        frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), bg_uint8, dtype=np.uint8)

        if pt_radius <= 1:
            # Single-pixel splat
            frame[v, u] = cols
        else:
            # Multi-pixel splat: repeat points with offsets for a square kernel
            offsets = []
            for dy in range(-pt_radius, pt_radius + 1):
                for dx in range(-pt_radius, pt_radius + 1):
                    if dx*dx + dy*dy <= pt_radius*pt_radius:  # circular kernel
                        offsets.append((dx, dy))
            
            all_u = np.concatenate([np.clip(u + dx, 0, VIDEO_WIDTH - 1) for dx, dy in offsets])
            all_v = np.concatenate([np.clip(v + dy, 0, VIDEO_HEIGHT - 1) for dx, dy in offsets])
            all_c = np.tile(cols, (len(offsets), 1))
            
            # Depth-sorted order is preserved: later writes (closer) overwrite earlier (farther)
            frame[all_v, all_u] = all_c

        writer.write(frame)

        pct = (frame_idx + 1) / n_frames * 100
        print(f"  Rendering: {pct:5.1f}%  (frame {frame_idx+1}/{n_frames})", end="\r")

    writer.release()
    print(f"\n\nVideo saved to:\n  {OUTPUT_VIDEO}")
    print(f"  File size: {os.path.getsize(OUTPUT_VIDEO) / 1024 / 1024:.1f} MB")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("VGGT 3D Reconstruction — Turntable Video Generator")
    print("=" * 60)
    print(f"  Video:   {VIDEO_NAME}")
    print(f"  Results: {RESULTS_DIR}")

    # 1. Load reconstruction
    depth, extrinsics, intrinsics, s1_indices = load_reconstruction()

    # 2. Load frame colors (fallback to depth-based if frames not found)
    print("\nLoading frame colors...")
    colors_per_frame = load_frame_colors(s1_indices)
    if colors_per_frame is None:
        print("  Using depth-based coloring (frames not available)")

    # 3. Build point cloud
    points, colors = build_point_cloud(depth, extrinsics, intrinsics, colors_per_frame)

    # 4. Render turntable video
    render_turntable(points, colors)

    print("\nDone!")


if __name__ == "__main__":
    main()