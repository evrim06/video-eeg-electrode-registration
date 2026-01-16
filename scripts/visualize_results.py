import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mne
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RESULTS_FILE = r"results/electrodes_3d.json"
SCANNER_FILE = r"Scanner_recordings/24ch_motor_1-6-2026_2-45 PM.elc"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_pipeline_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    points = {}
    for source in [data.get("landmarks", {}), data.get("electrodes", {})]:
        for k, v in source.items():
            points[k.upper()] = np.array(v["position"])
    return points

def parse_scanner_elc(path):
    points = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    coords = [float(p) for p in parts[-3:]]
                    label = parts[0].replace(':', '').upper()
                    points[label] = np.array(coords)
                except ValueError: continue
    return points

def align_points(source_dict, target_dict):
    """Align source to target using NAS/LPA/RPA (Procrustes)"""
    common = ["NAS", "LPA", "RPA"]
    src_pts, tgt_pts = [], []
    for lm in common:
        if lm in source_dict and lm in target_dict:
            src_pts.append(source_dict[lm])
            tgt_pts.append(target_dict[lm])
    
    if len(src_pts) < 3: return source_dict

    src = np.array(src_pts)
    tgt = np.array(tgt_pts)
    
    src_mean = src.mean(0)
    tgt_mean = tgt.mean(0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    
    scale = np.linalg.norm(tgt_c) / np.linalg.norm(src_c)
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    aligned = {}
    for k, v in source_dict.items():
        aligned[k] = scale * (v - src_mean) @ R.T + tgt_mean
    return aligned

def match_electrodes(p_aligned, s_data, max_dist_mm=35.0):
    """Match electrodes using Geometry + Distance Threshold"""
    landmarks = ["NAS", "LPA", "RPA", "INION"]
    p_elec_keys = [k for k in p_aligned.keys() if k not in landmarks]
    s_elec_keys = [k for k in s_data.keys() if k not in landmarks]
    
    p_pts = np.array([p_aligned[k] for k in p_elec_keys])
    s_pts = np.array([s_data[k] for k in s_elec_keys])
    
    dists = cdist(p_pts, s_pts)
    row_ind, col_ind = linear_sum_assignment(dists)
    
    matches = []
    dropped_video = []
    
    # 1. Add Landmarks
    for lm in ["NAS", "LPA", "RPA"]:
        if lm in p_aligned and lm in s_data:
            err = np.linalg.norm(p_aligned[lm] - s_data[lm])
            matches.append({"name": lm, "error": err, "pos": s_data[lm]})

    # 2. Add Electrodes
    for i, j in zip(row_ind, col_ind):
        dist = dists[i, j]
        s_name = s_elec_keys[j]
        if dist < max_dist_mm:
            matches.append({
                "name": s_name, 
                "error": dist,
                "pos": s_data[s_name]
            })
        else:
            dropped_video.append(p_elec_keys[i])

    print(f"\n[FILTERING] Matched: {len(matches)} | Dropped: {len(dropped_video)}")
    return matches

# ==============================================================================
# MAIN TOPOPLOT LOGIC
# ==============================================================================

def generate_rotated_topoplot(matches):
    print(f"Plotting error map for {len(matches)} electrodes...")
    
    names = [m["name"] for m in matches]
    errors = [m["error"] for m in matches]
    
    # --- GLOBAL ROTATION FIX ---
    # We apply this transformation to ALL matched points.
    # Scanner Format: X=Front, Y=Left, Z=Up
    # MNE Format:     X=Right, Y=Front, Z=Up
    #
    # Transformation:
    # New X (Right) = -Old Y (Left)
    # New Y (Front) =  Old X (Front)
    # New Z (Up)    =  Old Z (Up)
    
    positions = []
    for m in matches:
        old_pos = m["pos"] / 1000.0  # Convert mm to meters
        
        # Apply Rotation
        new_x = -old_pos[1]  # -Left = Right
        new_y =  old_pos[0]  # Front = Front
        new_z =  old_pos[2]  # Up = Up
        
        positions.append(np.array([new_x, new_y, new_z]))
    # ---------------------------
    
    # Create MNE Montage
    montage_pos = dict(zip(names, positions))
    montage = mne.channels.make_dig_montage(ch_pos=montage_pos, coord_frame='head')
    
    info = mne.create_info(ch_names=names, sfreq=1, ch_types='eeg')
    info.set_montage(montage)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    im, cn = mne.viz.plot_topomap(
        data=np.array(errors), 
        pos=info, 
        axes=ax, 
        names=names,
        show=False,
        cmap='Spectral_r', 
        vlim=(0, 20),
        contours=0,
        image_interp='cubic'
    )
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Localization Error (mm)')
    plt.title(f"Electrode Accuracy Map (n={len(matches)})", fontsize=16)
    
    # Save to a NEW filename so we don't look at the old one
    output_file = "final_corrected_topoplot.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[SUCCESS] Topoplot saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(SCANNER_FILE):
        print("Error: Files not found.")
    else:
        p_data = parse_pipeline_json(RESULTS_FILE)
        s_data = parse_scanner_elc(SCANNER_FILE)
        p_aligned = align_points(p_data, s_data)
        matches = match_electrodes(p_aligned, s_data)
        
        try:
            generate_rotated_topoplot(matches)
        except Exception as e:
            print(f"Topoplot failed: {e}")