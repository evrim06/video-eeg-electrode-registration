"""
VGGT-YOLO Bridge Script

Projects 2D electrode detections onto the 3D head surface.

Uses:
- Tragus-tragus line (LPA-RPA) for X-axis
- Nasion-inion axis for Y-axis
- Fits landmarks to head surface
- Projects electrodes onto 3D scalp

Output: Final 3D electrode coordinates in standard head coordinate system (mm)
"""

import os
import sys
import json
import pickle
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.optimize import minimize


# CONFIGURATION


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Inputs
TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
ALIGNED_FILE = os.path.join(RESULTS_DIR, "aligned_positions.json")
CROP_INFO_FILE = os.path.join(RESULTS_DIR, "crop_info.json")
RECON_FILE = os.path.join(RESULTS_DIR, "vggt_output", "reconstruction.npz")

# Outputs
OUTPUT_JSON = os.path.join(RESULTS_DIR, "electrodes_3d.json")
OUTPUT_PLY = os.path.join(RESULTS_DIR, "electrodes_3d.ply")

# Video path
VIDEO_PATH = os.path.join(BASE_DIR, "data", "IMG_2763.mp4")

# Landmark IDs (must match Script 1)
LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
LANDMARK_INION = 3
NUM_LANDMARKS = 4

# Standard head dimensions (mm) for scaling
STANDARD_EAR_TO_EAR = 150.0  # LPA to RPA
STANDARD_NAS_TO_INION = 180.0  # Nasion to Inion



# COORDINATE TRANSFORMATION


def get_video_info(video_path):
    """Get original video dimensions."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, total_frames


def transform_coords_to_vggt_space(u, v, orig_width, orig_height, vggt_size=518):
    """Transform coordinates from original video space to VGGT space."""
    scale = vggt_size / max(orig_width, orig_height)
    new_w = int(orig_width * scale)
    new_h = int(orig_height * scale)
    pad_w = (vggt_size - new_w) // 2
    pad_h = (vggt_size - new_h) // 2
    
    u_vggt = u * scale + pad_w
    v_vggt = v * scale + pad_h
    
    return u_vggt, v_vggt


def build_frame_mapping(tracking_frames, vggt_num_frames, total_video_frames, frame_skip):
    """Map tracking frame indices to VGGT frame indices."""
    vggt_video_frames = np.linspace(0, total_video_frames - 1, vggt_num_frames, dtype=int)
    
    mapping = {}
    for track_idx in tracking_frames:
        video_frame = track_idx * frame_skip
        distances = np.abs(vggt_video_frames - video_frame)
        closest_vggt_idx = np.argmin(distances)
        if distances[closest_vggt_idx] <= frame_skip * 2:
            mapping[track_idx] = closest_vggt_idx
    
    return mapping



# 3D PROJECTION


def unproject_point(u, v, depth_map, intrinsic, extrinsic):
    """Convert 2D pixel to 3D world coordinate."""
    H, W = depth_map.shape
    
    if not (0 <= u < W - 1 and 0 <= v < H - 1):
        return None
    
    # Bilinear interpolation for depth
    x0, y0 = int(u), int(v)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = u - x0, v - y0
    
    d00, d01 = depth_map[y0, x0], depth_map[y0, x1]
    d10, d11 = depth_map[y1, x0], depth_map[y1, x1]
    
    if any(d <= 0 or not np.isfinite(d) for d in [d00, d01, d10, d11]):
        z = depth_map[int(round(v)), int(round(u))]
        if z <= 0 or not np.isfinite(z):
            return None
    else:
        z = d00*(1-wx)*(1-wy) + d01*wx*(1-wy) + d10*(1-wx)*wy + d11*wx*wy
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    P_cam = np.array([x_cam, y_cam, z, 1.0])
    
    T_world_cam = np.linalg.inv(extrinsic)
    P_world = T_world_cam @ P_cam
    
    return P_world[:3]


def project_electrode_to_surface(electrode_id, tracking_data, frame_mapping,
                                  depths, intrinsics, extrinsics,
                                  orig_w, orig_h, point_cloud, kdtree):
    """
    Project electrode to 3D surface using multiple views.
    Also snaps to nearest surface point for accuracy.
    """
    points_3d = []
    
    for track_frame_idx, data in tracking_data.items():
        if electrode_id not in data:
            continue
        if track_frame_idx not in frame_mapping:
            continue
        
        vggt_idx = frame_mapping[track_frame_idx]
        u_orig, v_orig = data[electrode_id]
        u_vggt, v_vggt = transform_coords_to_vggt_space(u_orig, v_orig, orig_w, orig_h)
        
        p3d = unproject_point(u_vggt, v_vggt, depths[vggt_idx], 
                             intrinsics[vggt_idx], extrinsics[vggt_idx])
        
        if p3d is not None:
            points_3d.append(p3d)
    
    if len(points_3d) == 0:
        return None
    
    pts = np.array(points_3d)
    
    # Robust mean with outlier removal
    if len(pts) > 5:
        for _ in range(2):
            mean = np.mean(pts, axis=0)
            dists = np.linalg.norm(pts - mean, axis=1)
            threshold = np.mean(dists) + 2 * np.std(dists)
            mask = dists < threshold
            if np.sum(mask) >= 3:
                pts = pts[mask]
    
    initial_pos = np.mean(pts, axis=0)
    
    # Snap to nearest surface point
    dist, idx = kdtree.query(initial_pos, k=1)
    surface_pos = point_cloud[idx]
    
    return surface_pos, initial_pos, len(points_3d)



# HEAD COORDINATE SYSTEM


def define_head_coordinate_system(landmarks_3d):
    """
    Define head coordinate system using anatomical landmarks.
    
    Coordinate system:
    - Origin: Midpoint between LPA and RPA (ear-to-ear center)
    - X-axis: LPA -> RPA (left to right)
    - Y-axis: INION -> NAS (back to front, perpendicular to X)
    - Z-axis: Up (cross product of X and Y)
    
    Args:
        landmarks_3d: dict with keys LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA, LANDMARK_INION
    
    Returns:
        transform_info: dict with origin, rotation matrix, and scale factors
    """
    required = [LANDMARK_LPA, LANDMARK_RPA]
    if not all(lid in landmarks_3d for lid in required):
        print("  ⚠ Missing LPA or RPA! Cannot define coordinate system.")
        return None
    
    LPA = landmarks_3d[LANDMARK_LPA]
    RPA = landmarks_3d[LANDMARK_RPA]
    
    # Origin: midpoint between ears (tragus-tragus center)
    origin = (LPA + RPA) / 2
    
    # X-axis: LPA -> RPA (left to right)
    x_axis = RPA - LPA
    ear_to_ear = np.linalg.norm(x_axis)
    x_axis = x_axis / ear_to_ear
    
    # Determine Y direction from NAS and/or INION
    if LANDMARK_NAS in landmarks_3d and LANDMARK_INION in landmarks_3d:
        # Best case: have both
        NAS = landmarks_3d[LANDMARK_NAS]
        INION = landmarks_3d[LANDMARK_INION]
        y_dir = NAS - INION
        nas_to_inion = np.linalg.norm(y_dir)
    elif LANDMARK_NAS in landmarks_3d:
        NAS = landmarks_3d[LANDMARK_NAS]
        y_dir = NAS - origin
        nas_to_inion = np.linalg.norm(y_dir) * 2  # Estimate
    elif LANDMARK_INION in landmarks_3d:
        INION = landmarks_3d[LANDMARK_INION]
        y_dir = origin - INION
        nas_to_inion = np.linalg.norm(y_dir) * 2  # Estimate
    else:
        print(" Missing NAS and INION! Using estimated Y-axis.")
        # Estimate: perpendicular to X in horizontal plane
        y_dir = np.array([-x_axis[1], x_axis[0], 0])
        nas_to_inion = STANDARD_NAS_TO_INION * (ear_to_ear / STANDARD_EAR_TO_EAR)
    
    # Orthogonalize Y to X
    y_axis = y_dir - np.dot(y_dir, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Z-axis: up (cross product)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Ensure right-handed coordinate system
    if np.dot(z_axis, np.array([0, 0, 1])) < 0:
        z_axis = -z_axis
        y_axis = np.cross(z_axis, x_axis)
    
    # Rotation matrix (world to head)
    R = np.array([x_axis, y_axis, z_axis])
    
    # Scale to standard head size (mm)
    scale = STANDARD_EAR_TO_EAR / ear_to_ear
    
    return {
        "origin": origin,
        "rotation": R,
        "scale": scale,
        "ear_to_ear_raw": ear_to_ear,
        "nas_to_inion_raw": nas_to_inion,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis
    }


def transform_to_head_coordinates(position_3d, transform_info):
    """Transform a 3D point to head coordinate system."""
    if transform_info is None:
        return position_3d
    
    origin = transform_info["origin"]
    R = transform_info["rotation"]
    scale = transform_info["scale"]
    
    p_centered = position_3d - origin
    p_rotated = R @ p_centered
    p_scaled = p_rotated * scale
    
    return p_scaled



# SAVE RESULTS


def save_results(positions_3d, head_transform, projection_stats):
    """Save 3D electrode positions."""
    
    landmarks = {}
    electrodes = {}
    
    for eid, pos in positions_3d.items():
        pos_list = pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
        
        if eid == LANDMARK_NAS:
            landmarks["NAS"] = pos_list
        elif eid == LANDMARK_LPA:
            landmarks["LPA"] = pos_list
        elif eid == LANDMARK_RPA:
            landmarks["RPA"] = pos_list
        elif eid == LANDMARK_INION:
            landmarks["INION"] = pos_list
        else:
            electrodes[f"E{eid - NUM_LANDMARKS}"] = pos_list
    
    output = {
        "coordinate_system": {
            "type": "head_aligned_3d",
            "origin": "midpoint between LPA and RPA (tragus-tragus center)",
            "x_axis": "LPA to RPA (left to right)",
            "y_axis": "INION to NAS (back to front)",
            "z_axis": "inferior to superior (down to up)"
        },
        "units": "mm",
        "scale_reference": {
            "ear_to_ear": STANDARD_EAR_TO_EAR,
            "nas_to_inion": STANDARD_NAS_TO_INION
        },
        "landmarks": landmarks,
        "electrodes": electrodes,
        "num_electrodes": len(electrodes),
        "projection_stats": projection_stats,
        "head_transform": {
            "origin": head_transform["origin"].tolist() if head_transform else None,
            "scale": head_transform["scale"] if head_transform else 1.0,
            "ear_to_ear_raw": head_transform["ear_to_ear_raw"] if head_transform else None
        }
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✓ Saved JSON: {OUTPUT_JSON}")
    
    # Save PLY
    all_positions = list(positions_3d.values())
    all_ids = list(positions_3d.keys())
    
    with open(OUTPUT_PLY, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_positions)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for eid, pos in zip(all_ids, all_positions):
            if eid < NUM_LANDMARKS:
                r, g, b = 255, 0, 0  # Red for landmarks
            else:
                r, g, b = 0, 100, 255  # Blue for electrodes
            f.write(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {r} {g} {b}\n")
    
    print(f"✓ Saved PLY: {OUTPUT_PLY}")


def visualize_results(positions_3d, head_transform):
    """Matplotlib visualization of 3D electrode positions."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("  (matplotlib not available)")
        return
    
    fig = plt.figure(figsize=(16, 5))
    
    # Separate landmarks and electrodes
    landmarks = {k: v for k, v in positions_3d.items() if k < NUM_LANDMARKS}
    electrodes = {k: v for k, v in positions_3d.items() if k >= NUM_LANDMARKS}
    
    landmark_names = {LANDMARK_NAS: "NAS", LANDMARK_LPA: "LPA", 
                      LANDMARK_RPA: "RPA", LANDMARK_INION: "INION"}
    
    # Plot 1: Top view (X-Y)
    ax1 = fig.add_subplot(141)
    for eid, pos in electrodes.items():
        ax1.scatter(pos[0], pos[1], c='blue', s=50)
        ax1.annotate(f'E{eid-NUM_LANDMARKS}', (pos[0], pos[1]), fontsize=6)
    for eid, pos in landmarks.items():
        ax1.scatter(pos[0], pos[1], c='red', s=100, marker='s')
        ax1.annotate(landmark_names.get(eid, str(eid)), (pos[0]+2, pos[1]+2), fontsize=8, fontweight='bold')
    # Draw head outline
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(STANDARD_EAR_TO_EAR/2 * np.cos(theta), STANDARD_NAS_TO_INION/2 * np.sin(theta), 'k--', alpha=0.3)
    ax1.set_xlabel('X (Left-Right) mm')
    ax1.set_ylabel('Y (Back-Front) mm')
    ax1.set_title('Top View')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Front view (X-Z)
    ax2 = fig.add_subplot(142)
    for eid, pos in electrodes.items():
        ax2.scatter(pos[0], pos[2], c='blue', s=50)
    for eid, pos in landmarks.items():
        ax2.scatter(pos[0], pos[2], c='red', s=100, marker='s')
    ax2.set_xlabel('X (Left-Right) mm')
    ax2.set_ylabel('Z (Down-Up) mm')
    ax2.set_title('Front View')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Side view (Y-Z)
    ax3 = fig.add_subplot(143)
    for eid, pos in electrodes.items():
        ax3.scatter(pos[1], pos[2], c='blue', s=50)
    for eid, pos in landmarks.items():
        ax3.scatter(pos[1], pos[2], c='red', s=100, marker='s')
    ax3.set_xlabel('Y (Back-Front) mm')
    ax3.set_ylabel('Z (Down-Up) mm')
    ax3.set_title('Side View')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 3D view
    ax4 = fig.add_subplot(144, projection='3d')
    for eid, pos in electrodes.items():
        ax4.scatter(pos[0], pos[1], pos[2], c='blue', s=50)
    for eid, pos in landmarks.items():
        ax4.scatter(pos[0], pos[1], pos[2], c='red', s=100, marker='s')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('3D View')
    
    plt.tight_layout()
    
    fig_path = os.path.join(RESULTS_DIR, "electrodes_3d_visualization.png")
    plt.savefig(fig_path, dpi=150)
    print(f"✓ Saved visualization: {fig_path}")
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("BRIDGE: 2D TRACKING -> 3D ELECTRODE POSITIONS")
    print("=" * 70)
    
    # --- 1. LOAD DATA ---
    print("\n--- Step 1: Loading Data ---")
    
    if not os.path.exists(TRACKING_FILE):
        print(f"❌ Missing: {TRACKING_FILE}")
        sys.exit(1)
    
    if not os.path.exists(RECON_FILE):
        print(f"❌ Missing: {RECON_FILE}")
        sys.exit(1)
    
    with open(TRACKING_FILE, "rb") as f:
        tracking_data = pickle.load(f)
    print(f"  ✓ Tracking: {len(tracking_data)} frames")
    
    crop_info = {"x": 0, "y": 0, "skip": 10}
    if os.path.exists(CROP_INFO_FILE):
        with open(CROP_INFO_FILE, "r") as f:
            crop_info = json.load(f)
    
    recon = np.load(RECON_FILE)
    depths = recon["depth"]
    intrinsics = recon["intrinsics"]
    extrinsics = recon["extrinsics"]
    point_cloud = recon["points"]
    print(f"  ✓ Reconstruction: {depths.shape[0]} frames, {len(point_cloud):,} points")
    
    orig_w, orig_h, total_frames = get_video_info(VIDEO_PATH)
    print(f"  ✓ Video: {orig_w}x{orig_h}, {total_frames} frames")
    
    # Build KD-tree for surface snapping
    kdtree = cKDTree(point_cloud)
    
    # --- 2. FRAME MAPPING ---
    print("\n--- Step 2: Frame Mapping ---")
    
    tracking_frames = list(tracking_data.keys())
    frame_mapping = build_frame_mapping(tracking_frames, depths.shape[0], 
                                        total_frames, crop_info["skip"])
    print(f"  ✓ Mapped {len(frame_mapping)}/{len(tracking_frames)} frames")
    
    # --- 3. FIND ELECTRODES ---
    print("\n--- Step 3: Finding Electrodes ---")
    
    all_ids = set()
    for frame_data in tracking_data.values():
        all_ids.update(frame_data.keys())
    
    num_landmarks = sum(1 for eid in all_ids if eid < NUM_LANDMARKS)
    num_electrodes = len(all_ids) - num_landmarks
    print(f"  ✓ {num_landmarks} landmarks + {num_electrodes} electrodes")
    
    # --- 4. PROJECT TO 3D ---
    print("\n--- Step 4: Projecting to 3D Surface ---")
    
    raw_positions_3d = {}
    projection_stats = {}
    
    for eid in sorted(all_ids):
        result = project_electrode_to_surface(
            eid, tracking_data, frame_mapping,
            depths, intrinsics, extrinsics,
            orig_w, orig_h, point_cloud, kdtree
        )
        
        if result is not None:
            surface_pos, initial_pos, num_views = result
            raw_positions_3d[eid] = surface_pos
            
            # Label
            if eid < NUM_LANDMARKS:
                label = ["NAS", "LPA", "RPA"][eid]  # Only 3 landmarks
            else:
                label = f"E{eid - NUM_LANDMARKS}"
            
            projection_stats[label] = {"views": num_views}
            print(f"  {label}: {num_views} views -> ({surface_pos[0]:.2f}, {surface_pos[1]:.2f}, {surface_pos[2]:.2f})")
        else:
            if eid < NUM_LANDMARKS:
                label = ["NAS", "LPA", "RPA", "INION"][eid]
            else:
                label = f"E{eid - NUM_LANDMARKS}"
            print(f"  {label}: ⚠ No valid 3D projection!")
    
    # --- 5. DEFINE HEAD COORDINATE SYSTEM ---
    print("\n--- Step 5: Defining Head Coordinate System ---")
    
    head_transform = define_head_coordinate_system(raw_positions_3d)
    
    if head_transform:
        print(f"  ✓ Origin: Tragus-tragus midpoint")
        print(f"  ✓ Scale: {head_transform['scale']:.4f}")
        print(f"  ✓ Raw ear-to-ear: {head_transform['ear_to_ear_raw']:.2f}")
    
    # --- 6. TRANSFORM TO HEAD COORDINATES ---
    print("\n--- Step 6: Transforming to Head Coordinates ---")
    
    final_positions_3d = {}
    for eid, pos in raw_positions_3d.items():
        final_positions_3d[eid] = transform_to_head_coordinates(pos, head_transform)
    
    # --- 7. SAVE ---
    print("\n--- Step 7: Saving Results ---")
    save_results(final_positions_3d, head_transform, projection_stats)
    
    # --- 8. VISUALIZE ---
    print("\n--- Step 8: Visualization ---")
    visualize_results(final_positions_3d, head_transform)
    
    # --- DONE ---
    print("\n" + "=" * 70)
    print("BRIDGE COMPLETE!")
    print("=" * 70)
    print(f"\nFinal 3D positions:")
    print(f"  - JSON: {OUTPUT_JSON}")
    print(f"  - PLY:  {OUTPUT_PLY}")
    print(f"\n{len([k for k in final_positions_3d if k >= NUM_LANDMARKS])} electrodes + "
          f"{len([k for k in final_positions_3d if k < NUM_LANDMARKS])} landmarks")


if __name__ == "__main__":
    main()