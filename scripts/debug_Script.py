"""
DEBUG SCRIPT - Diagnose 3D Pipeline Issues
==========================================
Run this to understand what's going wrong.
"""

import os
import sys
import json
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
RECON_FILE = os.path.join(RESULTS_DIR, "vggt_output", "reconstruction.npz")
CROP_INFO_FILE = os.path.join(RESULTS_DIR, "crop_info.json")

LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2


def transform_coords_to_vggt_space(u, v, crop_w, crop_h, vggt_size=518):
    scale = vggt_size / max(crop_w, crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    pad_w, pad_h = (vggt_size - new_w) // 2, (vggt_size - new_h) // 2
    return u * scale + pad_w, v * scale + pad_h


def unproject_point(u, v, depth_map, intrinsic, extrinsic):
    H, W = depth_map.shape
    
    if not (0 <= u < W - 1 and 0 <= v < H - 1):
        return None
    
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
    P_world = np.linalg.inv(extrinsic) @ P_cam
    
    return P_world[:3]


def main():
    print("=" * 70)
    print("DEBUG: Diagnosing 3D Pipeline")
    print("=" * 70)
    
    # Load data
    with open(TRACKING_FILE, "rb") as f:
        tracking_data = pickle.load(f)
    
    with open(CROP_INFO_FILE, "r") as f:
        crop = json.load(f)
    
    recon = np.load(RECON_FILE)
    
    # Build frame mapping
    s1_to_vggt = {int(s1): int(v) for v, s1 in zip(
        recon["frame_mapping_keys"],
        recon["frame_mapping_values"]
    )}
    
    depth_maps = np.squeeze(recon["depth"])
    intrinsics = np.squeeze(recon["intrinsics"])
    extrinsics = np.squeeze(recon["extrinsics"])
    
    # Handle 3x4 -> 4x4
    if extrinsics.ndim == 3 and extrinsics.shape[-2:] == (3, 4):
        num_frames = extrinsics.shape[0]
        extr_4x4 = np.zeros((num_frames, 4, 4))
        for i in range(num_frames):
            extr_4x4[i] = np.eye(4)
            extr_4x4[i, :3, :] = extrinsics[i]
        extrinsics = extr_4x4
    
    print(f"\n--- Data Shapes ---")
    print(f"  Depth: {depth_maps.shape}")
    print(f"  Intrinsics: {intrinsics.shape}")
    print(f"  Extrinsics: {extrinsics.shape}")
    print(f"  Crop: {crop}")
    
    # =========================================================================
    # DEBUG 1: Check 2D tracking coordinates
    # =========================================================================
    print(f"\n--- DEBUG 1: 2D Tracking Coordinates ---")
    
    sample_frames = list(tracking_data.keys())[:3]
    for s1_idx in sample_frames:
        tracks = tracking_data[s1_idx]
        print(f"\n  Frame {s1_idx}:")
        for eid, (u, v) in sorted(tracks.items())[:5]:
            label = ["NAS", "LPA", "RPA"][eid] if eid < 3 else f"E{eid-3}"
            print(f"    {label}: 2D = ({u:.1f}, {v:.1f})")
    
    # =========================================================================
    # DEBUG 2: Check depth values at landmark locations
    # =========================================================================
    print(f"\n--- DEBUG 2: Depth Values at Landmarks ---")
    
    for s1_idx in sample_frames:
        if s1_idx not in s1_to_vggt:
            print(f"\n  Frame {s1_idx}: No VGGT mapping!")
            continue
        
        vggt_idx = s1_to_vggt[s1_idx]
        depth = depth_maps[vggt_idx]
        tracks = tracking_data[s1_idx]
        
        print(f"\n  Frame {s1_idx} (VGGT idx {vggt_idx}):")
        print(f"    Depth map: min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}")
        
        for eid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
            if eid not in tracks:
                print(f"    {['NAS','LPA','RPA'][eid]}: NOT TRACKED!")
                continue
            
            u, v = tracks[eid]
            u_vggt, v_vggt = transform_coords_to_vggt_space(u, v, crop["w"], crop["h"])
            
            # Get depth at this pixel
            u_int, v_int = int(u_vggt), int(v_vggt)
            if 0 <= u_int < 518 and 0 <= v_int < 518:
                z = depth[v_int, u_int]
            else:
                z = np.nan
            
            label = ["NAS", "LPA", "RPA"][eid]
            print(f"    {label}: 2D=({u:.1f}, {v:.1f}) → VGGT=({u_vggt:.1f}, {v_vggt:.1f}) → depth={z:.4f}")
    
    # =========================================================================
    # DEBUG 3: Check camera intrinsics
    # =========================================================================
    print(f"\n--- DEBUG 3: Camera Intrinsics (Frame 0) ---")
    intr = intrinsics[0]
    print(f"  [[{intr[0,0]:.2f}, {intr[0,1]:.2f}, {intr[0,2]:.2f}],")
    print(f"   [{intr[1,0]:.2f}, {intr[1,1]:.2f}, {intr[1,2]:.2f}],")
    print(f"   [{intr[2,0]:.2f}, {intr[2,1]:.2f}, {intr[2,2]:.2f}]]")
    print(f"  fx={intr[0,0]:.2f}, fy={intr[1,1]:.2f}, cx={intr[0,2]:.2f}, cy={intr[1,2]:.2f}")
    
    # =========================================================================
    # DEBUG 4: Check camera extrinsics
    # =========================================================================
    print(f"\n--- DEBUG 4: Camera Extrinsics (First 3 frames) ---")
    for i in range(min(3, len(extrinsics))):
        extr = extrinsics[i]
        # Extract rotation and translation
        R = extr[:3, :3]
        t = extr[:3, 3]
        print(f"\n  Frame {i}:")
        print(f"    Translation: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
        print(f"    Rotation det: {np.linalg.det(R):.4f} (should be ~1)")
    
    # =========================================================================
    # DEBUG 5: Unproject landmarks and check 3D positions
    # =========================================================================
    print(f"\n--- DEBUG 5: Unprojected 3D Landmark Positions ---")
    
    all_nas, all_lpa, all_rpa = [], [], []
    
    for s1_idx, tracks in tracking_data.items():
        if s1_idx not in s1_to_vggt:
            continue
        
        vggt_idx = s1_to_vggt[s1_idx]
        depth = depth_maps[vggt_idx]
        intr = intrinsics[vggt_idx]
        extr = extrinsics[vggt_idx]
        
        landmarks_3d = {}
        for eid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
            if eid not in tracks:
                continue
            u, v = tracks[eid]
            u_vggt, v_vggt = transform_coords_to_vggt_space(u, v, crop["w"], crop["h"])
            p3d = unproject_point(u_vggt, v_vggt, depth, intr, extr)
            if p3d is not None:
                landmarks_3d[eid] = p3d
        
        if all(eid in landmarks_3d for eid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]):
            all_nas.append(landmarks_3d[LANDMARK_NAS])
            all_lpa.append(landmarks_3d[LANDMARK_LPA])
            all_rpa.append(landmarks_3d[LANDMARK_RPA])
    
    print(f"\n  Frames with all 3 landmarks: {len(all_nas)}")
    
    if len(all_nas) > 0:
        all_nas = np.array(all_nas)
        all_lpa = np.array(all_lpa)
        all_rpa = np.array(all_rpa)
        
        print(f"\n  NAS (world coords):")
        print(f"    Mean: ({all_nas[:,0].mean():.4f}, {all_nas[:,1].mean():.4f}, {all_nas[:,2].mean():.4f})")
        print(f"    Std:  ({all_nas[:,0].std():.4f}, {all_nas[:,1].std():.4f}, {all_nas[:,2].std():.4f})")
        
        print(f"\n  LPA (world coords):")
        print(f"    Mean: ({all_lpa[:,0].mean():.4f}, {all_lpa[:,1].mean():.4f}, {all_lpa[:,2].mean():.4f})")
        print(f"    Std:  ({all_lpa[:,0].std():.4f}, {all_lpa[:,1].std():.4f}, {all_lpa[:,2].std():.4f})")
        
        print(f"\n  RPA (world coords):")
        print(f"    Mean: ({all_rpa[:,0].mean():.4f}, {all_rpa[:,1].mean():.4f}, {all_rpa[:,2].mean():.4f})")
        print(f"    Std:  ({all_rpa[:,0].std():.4f}, {all_rpa[:,1].std():.4f}, {all_rpa[:,2].std():.4f})")
        
        # Check distances between landmarks
        ear_dists = np.linalg.norm(all_rpa - all_lpa, axis=1)
        nas_to_origin = np.linalg.norm(all_nas - (all_lpa + all_rpa) / 2, axis=1)
        
        print(f"\n  LPA-RPA distances:")
        print(f"    Mean: {ear_dists.mean():.4f}")
        print(f"    Std:  {ear_dists.std():.4f}")
        print(f"    Min:  {ear_dists.min():.4f}")
        print(f"    Max:  {ear_dists.max():.4f}")
        
        print(f"\n  NAS to ear-center distances:")
        print(f"    Mean: {nas_to_origin.mean():.4f}")
        print(f"    Std:  {nas_to_origin.std():.4f}")
        
        # =====================================================================
        # DEBUG 6: Per-frame head transform test
        # =====================================================================
        print(f"\n--- DEBUG 6: Per-Frame Head Transform Test ---")
        
        # Take first frame with all landmarks
        nas = all_nas[0]
        lpa = all_lpa[0]
        rpa = all_rpa[0]
        
        print(f"\n  Frame 0 landmarks (world):")
        print(f"    NAS: ({nas[0]:.4f}, {nas[1]:.4f}, {nas[2]:.4f})")
        print(f"    LPA: ({lpa[0]:.4f}, {lpa[1]:.4f}, {lpa[2]:.4f})")
        print(f"    RPA: ({rpa[0]:.4f}, {rpa[1]:.4f}, {rpa[2]:.4f})")
        
        # Compute transform
        origin = (lpa + rpa) / 2.0
        x_axis = rpa - lpa
        ear_dist = np.linalg.norm(x_axis)
        x_axis = x_axis / ear_dist
        
        # Estimate inion
        nas_vec = nas - origin
        forward_dir = nas_vec - np.dot(nas_vec, x_axis) * x_axis
        forward_len = np.linalg.norm(forward_dir)
        forward_dir = forward_dir / forward_len
        inion = origin - forward_dir * forward_len
        
        y_dir = nas - inion
        y_axis = y_dir - np.dot(y_dir, x_axis) * x_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        R = np.array([x_axis, y_axis, z_axis])
        
        print(f"\n  Computed transform:")
        print(f"    Origin: ({origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f})")
        print(f"    Ear distance: {ear_dist:.4f}")
        print(f"    X-axis: ({x_axis[0]:.4f}, {x_axis[1]:.4f}, {x_axis[2]:.4f})")
        print(f"    Y-axis: ({y_axis[0]:.4f}, {y_axis[1]:.4f}, {y_axis[2]:.4f})")
        print(f"    Z-axis: ({z_axis[0]:.4f}, {z_axis[1]:.4f}, {z_axis[2]:.4f})")
        
        # Transform landmarks
        def transform(p):
            p_centered = p - origin
            p_rotated = R @ p_centered
            p_normalized = p_rotated / ear_dist
            return p_normalized
        
        nas_head = transform(nas)
        lpa_head = transform(lpa)
        rpa_head = transform(rpa)
        inion_head = transform(inion)
        
        print(f"\n  Landmarks in HEAD coords (normalized, ear=1):")
        print(f"    NAS:   ({nas_head[0]:.4f}, {nas_head[1]:.4f}, {nas_head[2]:.4f})")
        print(f"    LPA:   ({lpa_head[0]:.4f}, {lpa_head[1]:.4f}, {lpa_head[2]:.4f})  ← Should be (-0.5, ~0, ~0)")
        print(f"    RPA:   ({rpa_head[0]:.4f}, {rpa_head[1]:.4f}, {rpa_head[2]:.4f})  ← Should be (+0.5, ~0, ~0)")
        print(f"    INION: ({inion_head[0]:.4f}, {inion_head[1]:.4f}, {inion_head[2]:.4f})")
        
        # Scale to 150mm
        scale = 150.0
        print(f"\n  Landmarks in HEAD coords (mm, ear=150mm):")
        print(f"    NAS:   ({nas_head[0]*scale:.1f}, {nas_head[1]*scale:.1f}, {nas_head[2]*scale:.1f})")
        print(f"    LPA:   ({lpa_head[0]*scale:.1f}, {lpa_head[1]*scale:.1f}, {lpa_head[2]*scale:.1f})  ← Should be (-75, ~0, ~0)")
        print(f"    RPA:   ({rpa_head[0]*scale:.1f}, {rpa_head[1]*scale:.1f}, {rpa_head[2]*scale:.1f})  ← Should be (+75, ~0, ~0)")
        print(f"    INION: ({inion_head[0]*scale:.1f}, {inion_head[1]*scale:.1f}, {inion_head[2]*scale:.1f})")
        
        # Check ear distance after transform
        lpa_rpa_dist = np.linalg.norm(rpa_head - lpa_head)
        print(f"\n  LPA-RPA distance after transform: {lpa_rpa_dist:.4f} (should be 1.0)")
        print(f"  LPA-RPA distance in mm: {lpa_rpa_dist * scale:.1f} (should be 150.0)")


if __name__ == "__main__":
    main()
