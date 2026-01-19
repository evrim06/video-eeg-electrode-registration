import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

# Try importing MNE for topoplot, but don't crash if missing
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# ==============================================================================
# 1. PARSING & LOADING
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
    if not os.path.exists(path):
        return points
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        parts = line.split()
        if len(parts) >= 3:
            try:
                # Parse last 3 as coordinates
                coords = [float(p) for p in parts[-3:]]
                # Label is everything before
                label = parts[0].replace(':', '').upper()
                points[label] = np.array(coords)
            except ValueError:
                continue
    return points

# ==============================================================================
# 2. ALIGNMENT LOGIC (Procrustes)
# ==============================================================================

def align_points(source_dict, target_dict, used_landmarks=None):
    """
    Align source to target using specific landmarks.
    """
    if used_landmarks is None:
        used_landmarks = ["NAS", "LPA", "RPA"]

    src_pts, tgt_pts = [], []
    for lm in used_landmarks:
        if lm in source_dict and lm in target_dict:
            src_pts.append(source_dict[lm])
            tgt_pts.append(target_dict[lm])
    
    # Need at least 3 points for 3D alignment
    if len(src_pts) < 3: 
        # Fallback: Try adding INION if available
        if "INION" in source_dict and "INION" in target_dict:
            src_pts.append(source_dict["INION"])
            tgt_pts.append(target_dict["INION"])
    
    if len(src_pts) < 3:
        return source_dict, None # Cannot align

    src = np.array(src_pts)
    tgt = np.array(tgt_pts)
    
    # Center
    src_mean = src.mean(0)
    tgt_mean = tgt.mean(0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    
    # Scale
    scale = np.linalg.norm(tgt_c) / np.linalg.norm(src_c)
    
    # Rotate (Kabsch Algorithm)
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Apply to all
    aligned = {}
    for k, v in source_dict.items():
        aligned[k] = scale * (v - src_mean) @ R.T + tgt_mean
        
    return aligned, (scale, R, tgt_mean)

# ==============================================================================
# 3. METRICS: Nearest Neighbor & Regions
# ==============================================================================

def compute_nn_metrics(p_aligned, s_data):
    """
    Compute Nearest Neighbor error for every pipeline electrode.
    Does NOT assume labels match. Just finds closest physical point.
    """
    landmarks = ["NAS", "LPA", "RPA", "INION"]
    
    # Get lists of coordinates
    p_keys = [k for k in p_aligned.keys() if k not in landmarks]
    s_keys = [k for k in s_data.keys() if k not in landmarks]
    
    if not p_keys or not s_keys:
        return None

    p_pts = np.array([p_aligned[k] for k in p_keys])
    s_pts = np.array([s_data[k] for k in s_keys])
    
    # Build KDTree for Ground Truth
    tree = KDTree(s_pts)
    
    # Query every pipeline point against the tree
    distances, indices = tree.query(p_pts)
    
    # Create results list
    metrics = []
    for i, dist in enumerate(distances):
        # Determine Region based on ROTATED coordinate (MNE Frame: Y=Front, X=Right)
        # Note: p_pts here are in SCANNER frame (X=Front, Y=Left)
        # We need to map to regions based on Scanner frame logic
        
        pos = p_pts[i]
        x, y, z = pos[0], pos[1], pos[2] # Scanner: X=Front, Y=Left
        
        region = "Central"
        if x > 40: region = "Frontal"
        elif x < -40: region = "Occipital"
        elif y > 40: region = "Left"    # Scanner Y is Left
        elif y < -40: region = "Right"  # Scanner -Y is Right
        
        metrics.append({
            "label_video": p_keys[i],
            "nearest_scanner_idx": int(indices[i]),
            "error_mm": dist,
            "region": region,
            "pos_scanner_frame": pos
        })
        
    return metrics

def run_loo_test(p_data, s_data):
    """
    Leave-One-Out (LOO) Stability Test.
    Aligns 3 times, removing one landmark each time.
    Returns: Max drift (mm) observed.
    """
    landmarks = ["NAS", "LPA", "RPA", "INION"]
    available = [lm for lm in landmarks if lm in p_data and lm in s_data]
    
    if len(available) < 4:
        return None # Need 4 points to do LOO on 3-point sets safely
        
    baseline_aligned, _ = align_points(p_data, s_data, available)
    base_pts = np.array([v for k,v in baseline_aligned.items() if k not in landmarks])
    
    drifts = []
    
    # Try removing NAS, then LPA, then RPA
    for remove_lm in ["NAS", "LPA", "RPA"]:
        subset = [lm for lm in available if lm != remove_lm]
        if len(subset) < 3: continue
        
        # Re-align with subset
        loo_aligned, _ = align_points(p_data, s_data, subset)
        loo_pts = np.array([v for k,v in loo_aligned.items() if k not in landmarks])
        
        # Measure shift
        drift = np.linalg.norm(base_pts - loo_pts, axis=1).mean()
        drifts.append(drift)
        
    return np.max(drifts) if drifts else 0.0

# ==============================================================================
# 4. PLOTTING: Distribution & Topomap
# ==============================================================================

def plot_distributions(metrics, output_dir):
    errors = [m["error_mm"] for m in metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Histogram
    axes[0].hist(errors, bins=10, color='skyblue', edgecolor='black')
    axes[0].set_title("Error Distribution (Histogram)")
    axes[0].set_xlabel("Error (mm)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.1f}mm')
    axes[0].legend()
    
    # 2. CDF
    sorted_err = np.sort(errors)
    yvals = np.arange(len(sorted_err)) / float(len(sorted_err) - 1)
    axes[1].plot(sorted_err, yvals, linewidth=2, color='green')
    axes[1].set_title("Cumulative Distribution Function (CDF)")
    axes[1].set_xlabel("Error threshold (mm)")
    axes[1].set_ylabel("Fraction of Electrodes")
    axes[1].grid(True)
    
    # 3. Boxplot by Region
    regions = {}
    for m in metrics:
        r = m["region"]
        regions.setdefault(r, []).append(m["error_mm"])
    
    labels = list(regions.keys())
    data = [regions[l] for l in labels]
    
    axes[2].boxplot(data, labels=labels, patch_artist=True)
    axes[2].set_title("Error by Region")
    axes[2].set_ylabel("Error (mm)")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "distribution_plots.png")
    plt.savefig(save_path, dpi=150)
    print(f" Saved distributions to: {save_path}")

def plot_unlabeled_topomap(metrics, output_dir):
    if not HAS_MNE:
        print(" Skipping Topomap (MNE not installed)")
        return

    # Prepare data for MNE
    errors = []
    positions = []
    names = [] # Dummy names needed for MNE
    
    for i, m in enumerate(metrics):
        errors.append(m["error_mm"])
        # ROTATION FIX: Scanner(X,Y) -> MNE(Y,-X)
        raw = m["pos_scanner_frame"]
        # Rotate: New X = -Old Y, New Y = Old X
        rotated_pos = np.array([-raw[1], raw[0], raw[2]]) / 1000.0 # to meters
        positions.append(rotated_pos)
        names.append(str(i)) # ID doesn't matter, we won't show labels

    # Create Montage
    montage_pos = dict(zip(names, positions))
    montage = mne.channels.make_dig_montage(ch_pos=montage_pos, coord_frame='head')
    info = mne.create_info(ch_names=names, sfreq=1, ch_types='eeg')
    info.set_montage(montage)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im, _ = mne.viz.plot_topomap(
        data=np.array(errors),
        pos=info,
        axes=ax,
        names=None,          # Hides labels!
        show=False,
        cmap='Spectral_r',
        vlim=(0, 25),
        contours=0,
        image_interp='cubic'
    )
    
    # Add caption instead of labels
    plt.title("Nearest-Neighbor Error Projection (Unlabeled)", fontsize=16)
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Proximity Error (mm)")
    plt.figtext(0.5, 0.05, "Color represents distance to nearest ground-truth electrode.", 
                ha="center", fontsize=10, style='italic')
    
    save_path = os.path.join(output_dir, "error_projection_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" Saved error map to: {save_path}")

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pipeline", required=False, help="Pipeline JSON")
    parser.add_argument("-g", "--ground_truth", required=False, help="Scanner .elc")
    parser.add_argument("-o", "--output", default="results", help="Output folder")
    args = parser.parse_args()

    # --- 1. Auto-Resolution of Files ---
    p_file = args.pipeline or "results/IMG_3841/electrodes_3d.json"
    g_file = args.ground_truth or "Scanner_recordings/24ch_motor_1-6-2026_2-45 PM.elc"
    
    if not os.path.exists(p_file) or not os.path.exists(g_file):
        print("Error: Could not find files.")
        return

    print(f"\n--- SCRIPT 3: QUANTITATIVE EVALUATION ---")
    print(f"Pipeline: {p_file}")
    print(f"Scanner:  {g_file}")
    
    # --- 2. Load & Align ---
    p_data = parse_pipeline_json(p_file)
    s_data = parse_scanner_elc(g_file)
    
    if len(p_data) == 0 or len(s_data) == 0:
        print("Error: Empty data files.")
        return

    print("Aligning using Landmarks (NAS, LPA, RPA)...")
    p_aligned, _ = align_points(p_data, s_data)
    
    if not p_aligned:
        print("Alignment failed (missing landmarks).")
        return

    # --- 3. Compute Metrics (The New Way) ---
    print("Computing Nearest-Neighbor Metrics...")
    metrics = compute_nn_metrics(p_aligned, s_data)
    
    if not metrics:
        print("No electrodes found to compare.")
        return

    errors = [m["error_mm"] for m in metrics]
    
    # --- 4. Statistics ---
    print("\n" + "="*30)
    print("RESULTS SUMMARY (Nearest Neighbor)")
    print("="*30)
    print(f"Mean Error:   {np.mean(errors):.2f} mm")
    print(f"Median Error: {np.median(errors):.2f} mm")
    print(f"Std Dev:      {np.std(errors):.2f} mm")
    print(f"90th %ile:    {np.percentile(errors, 90):.2f} mm")
    print("-" * 30)
    
    # Region Stats
    regions = {}
    for m in metrics:
        regions.setdefault(m["region"], []).append(m["error_mm"])
    
    print("Regional Accuracy:")
    for r, errs in regions.items():
        print(f"  {r:10s}: {np.mean(errs):.2f} mm (n={len(errs)})")
        
    # --- 5. LOO Test ---
    print("\nRunning Leave-One-Out Stability Test...")
    loo_drift = run_loo_test(p_data, s_data)
    if loo_drift is not None:
        print(f"  Landmark Stability Drift: {loo_drift:.2f} mm")
    else:
        print("  Skipped (Not enough landmarks for LOO)")

    # --- 6. Visualization ---
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    print("\nGenerating visualizations...")
    plot_distributions(metrics, args.output)
    plot_unlabeled_topomap(metrics, args.output)
    
    print("\n[DONE] Evaluation complete.")

if __name__ == "__main__":
    main()