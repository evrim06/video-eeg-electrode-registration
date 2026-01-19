
"""
Script 3: Comprehensive Statistical Analysis & Visualization
EEG Electrode Localization Pipeline Validation

Features:
- ICP alignment (Clausner 2017, Taberna 2019)
- LA<5mm metric (Taberna 2019)
- Median + MAD statistics (Mazzonetto 2022)
- Wilcoxon test (comparing methods)
- Discrete electrode error visualization (no interpolation)
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.stats import wilcoxon
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import PatchCollection

# Optional: MNE for anatomical head outline
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# PARSING & LOADING

def parse_pipeline_json(path):
    """Load pipeline output JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    points = {}
    for source in [data.get("landmarks", {}), data.get("electrodes", {})]:
        for k, v in source.items():
            points[k.upper()] = np.array(v["position"])
    
    return points


def parse_scanner_elc(path):
    """Load 3D scanner ELC file."""
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
                coords = [float(p) for p in parts[-3:]]
                label = parts[0].replace(':', '').upper()
                points[label] = np.array(coords)
            except ValueError:
                continue
    
    return points


# ALIGNMENT: ICP (Iterative Closest Point)

def icp_alignment(source, target, max_iterations=50, tolerance=0.001):
    """
    ICP alignment following Clausner (2017) and Taberna (2019).
    
    Args:
        source: Nx3 array (video-based electrodes)
        target: Mx3 array (scanner electrodes)
        max_iterations: Maximum ICP iterations
        tolerance: Convergence threshold (mm)
    
    Returns:
        aligned_source: Transformed source points
        transformation: (scale, rotation, translation)
        final_error: RMS error after alignment
    """
    from scipy.spatial import cKDTree
    
    current = source.copy()
    prev_error = np.inf
    
    for iteration in range(max_iterations):
        # Find nearest neighbors
        tree = cKDTree(target)
        distances, indices = tree.query(current)
        
        # Matched pairs
        matched_target = target[indices]
        
        # Compute transformation (Procrustes)
        mtx1, mtx2, disparity = procrustes(matched_target, current)
        
        # Apply transformation
        # Procrustes returns standardized matrices, we need the actual transform
        # So we compute it manually:
        
        # Center
        source_mean = current.mean(axis=0)
        target_mean = matched_target.mean(axis=0)
        
        current_c = current - source_mean
        target_c = matched_target - target_mean
        
        # Scale
        scale = np.linalg.norm(target_c) / np.linalg.norm(current_c)
        
        # Rotation (Kabsch)
        H = current_c.T @ target_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply transform
        current = scale * (current - source_mean) @ R.T + target_mean
        
        # Check convergence
        error = np.mean(distances)
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
    
    transformation = {
        'scale': scale,
        'rotation': R,
        'translation': target_mean - scale * source_mean @ R.T,
        'iterations': iteration + 1
    }
    
    return current, transformation, error


def landmark_based_alignment(source_dict, target_dict, landmarks=["NAS", "LPA", "RPA"]):
    """
    Fallback: Rigid alignment using anatomical landmarks.
    Used if ICP fails or as initial guess.
    """
    src_pts, tgt_pts = [], []
    
    for lm in landmarks:
        if lm in source_dict and lm in target_dict:
            src_pts.append(source_dict[lm])
            tgt_pts.append(target_dict[lm])
    
    if len(src_pts) < 3:
        return None, None
    
    src = np.array(src_pts)
    tgt = np.array(tgt_pts)
    
    # Procrustes
    src_mean = src.mean(0)
    tgt_mean = tgt.mean(0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    
    scale = np.linalg.norm(tgt_c) / np.linalg.norm(src_c)
    
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Transform all points
    aligned = {}
    for k, v in source_dict.items():
        aligned[k] = scale * (v - src_mean) @ R.T + tgt_mean
    
    return aligned, (scale, R, tgt_mean)


# METRICS & STATISTICS


def compute_nearest_neighbor_errors(pipeline_points, scanner_points):
    """
    Compute nearest-neighbor distances (no label matching required).
    
    Returns:
        errors: List of {label, error_mm, position, region}
    """
    from scipy.spatial import cKDTree
    
    # Separate landmarks from electrodes
    landmarks = ["NAS", "LPA", "RPA", "INION"]
    
    p_labels = [k for k in pipeline_points.keys() if k not in landmarks]
    s_labels = [k for k in scanner_points.keys() if k not in landmarks]
    
    if not p_labels or not s_labels:
        return []
    
    p_coords = np.array([pipeline_points[k] for k in p_labels])
    s_coords = np.array([scanner_points[k] for k in s_labels])
    
    # Build KDTree
    tree = cKDTree(s_coords)
    distances, indices = tree.query(p_coords)
    
    # Assign regions (based on scanner coordinate system)
    results = []
    for i, (label, dist) in enumerate(zip(p_labels, distances)):
        pos = p_coords[i]
        x, y, z = pos
        
        # Region classification (adjust based on your coordinate system)
        if x > 40:
            region = "Frontal"
        elif x < -40:
            region = "Occipital"
        elif y > 40:
            region = "Left"
        elif y < -40:
            region = "Right"
        else:
            region = "Central"
        
        results.append({
            "label": label,
            "error_mm": dist,
            "position": pos,
            "region": region,
            "matched_scanner_label": s_labels[indices[i]]
        })
    
    return results


def compute_statistics(errors):
    """
    Compute comprehensive statistics following literature standards.
    
    Returns dict with:
        - Mean, Median, SD, MAD (Mazzonetto 2022)
        - Percentiles (Taberna 2019)
        - LA<5mm (Taberna 2019)
    """
    errors_array = np.array([e["error_mm"] for e in errors])
    
    stats = {
        # Central tendency
        "mean": np.mean(errors_array),
        "median": np.median(errors_array),
        
        # Dispersion
        "std": np.std(errors_array, ddof=1),  # Sample SD
        "mad": np.median(np.abs(errors_array - np.median(errors_array))),
        
        # Range
        "min": np.min(errors_array),
        "max": np.max(errors_array),
        
        # Percentiles
        "p25": np.percentile(errors_array, 25),
        "p75": np.percentile(errors_array, 75),
        "p90": np.percentile(errors_array, 90),
        "p95": np.percentile(errors_array, 95),
        
        # Localization Accuracy (Taberna 2019)
        "LA_5mm": np.sum(errors_array < 5.0) / len(errors_array) * 100,
        "LA_10mm": np.sum(errors_array < 10.0) / len(errors_array) * 100,
        
        # Sample size
        "n": len(errors_array)
    }
    
    return stats


def regional_statistics(errors):
    """Compute per-region statistics."""
    regions = {}
    for e in errors:
        r = e["region"]
        if r not in regions:
            regions[r] = []
        regions[r].append(e["error_mm"])
    
    regional_stats = {}
    for region, errs in regions.items():
        regional_stats[region] = {
            "n": len(errs),
            "mean": np.mean(errs),
            "median": np.median(errs),
            "std": np.std(errs, ddof=1) if len(errs) > 1 else 0,
            "mad": np.median(np.abs(np.array(errs) - np.median(errs)))
        }
    
    return regional_stats


# VISUALIZATION: DISCRETE ELECTRODE ERRORS

def plot_discrete_electrode_errors(errors, output_dir):
    """
    Plot errors as discrete colored circles at electrode positions.
    NO INTERPOLATION - following your correct observation!
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Extract data
    positions = np.array([e["position"] for e in errors])
    error_values = np.array([e["error_mm"] for e in errors])
    labels = [e["label"] for e in errors]
    
    # Color mapping
    vmin, vmax = 0, 15  # Error range in mm
    cmap = plt.cm.Spectral_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    #PLOT 1: Top View (X-Y plane)
    ax1.set_aspect('equal')
    
    # Draw head outline (circle)
    head_circle = Circle((0, 0), 100, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(head_circle)
    
    # Nose indicator (triangle at front)
    nose = plt.Polygon([[0, 100], [-10, 110], [10, 110]], 
                       closed=True, facecolor='black')
    ax1.add_patch(nose)
    
    # Ears
    left_ear = Ellipse((-100, 0), 20, 40, fill=False, edgecolor='black', linewidth=1.5)
    right_ear = Ellipse((100, 0), 20, 40, fill=False, edgecolor='black', linewidth=1.5)
    ax1.add_patch(left_ear)
    ax1.add_patch(right_ear)
    
    # Plot electrodes as discrete colored circles
    for i, (pos, err, label) in enumerate(zip(positions, error_values, labels)):
        x, y = pos[1], pos[0]  # Swap for proper orientation
        color = cmap(norm(err))
        
        # Draw circle
        circle = Circle((x, y), 8, facecolor=color, edgecolor='black', 
                       linewidth=0.5, zorder=10)
        ax1.add_patch(circle)
        
        # Optional: Add label (comment out if too crowded)
        # ax1.text(x, y, label, fontsize=6, ha='center', va='center')
    
    ax1.set_xlim(-120, 120)
    ax1.set_ylim(-120, 120)
    ax1.set_xlabel('Left ← → Right (mm)', fontsize=12)
    ax1.set_ylabel('Back ← → Front (mm)', fontsize=12)
    ax1.set_title('Electrode Localization Error (Top View)\nDiscrete Markers - No Interpolation', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    #PLOT 2: Side View (X-Z plane)
    ax2.set_aspect('equal')
    
    # Draw head outline (circle)
    head_circle2 = Circle((0, 0), 100, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(head_circle2)
    
    # Nose (at front, pointing right)
    nose2 = plt.Polygon([[100, 0], [110, -10], [110, 10]], 
                        closed=True, facecolor='black')
    ax2.add_patch(nose2)
    
    # Plot electrodes
    for i, (pos, err, label) in enumerate(zip(positions, error_values, labels)):
        x, z = pos[0], pos[2]  # Front-back vs up-down
        color = cmap(norm(err))
        
        circle = Circle((x, z), 8, facecolor=color, edgecolor='black', 
                       linewidth=0.5, zorder=10)
        ax2.add_patch(circle)
    
    ax2.set_xlim(-120, 120)
    ax2.set_ylim(-120, 120)
    ax2.set_xlabel('Back ← → Front (mm)', fontsize=12)
    ax2.set_ylabel('Bottom ← → Top (mm)', fontsize=12)
    ax2.set_title('Electrode Localization Error (Side View)\nDiscrete Markers - No Interpolation', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    #Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label('Localization Error (mm)', fontsize=12)
    
    # Add caption
    fig.text(0.5, 0.02, 
             'Each circle = one electrode. Color = nearest-neighbor error. No spatial interpolation.',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    save_path = os.path.join(output_dir, "discrete_electrode_errors.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved discrete error plot: {save_path}")
    plt.close()


def plot_error_distributions(errors, stats, output_dir):
    """Plot comprehensive error distributions."""
    error_values = np.array([e["error_mm"] for e in errors])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    #1. Histogram with statistics
    ax = axes[0, 0]
    ax.hist(error_values, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {stats["mean"]:.2f} mm')
    ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2,
               label=f'Median: {stats["median"]:.2f} mm')
    ax.axvline(5.0, color='orange', linestyle=':', linewidth=2,
               label='5mm threshold')
    ax.set_xlabel('Error (mm)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========== 2. CDF ==========
    ax = axes[0, 1]
    sorted_errors = np.sort(error_values)
    yvals = np.arange(len(sorted_errors)) / (len(sorted_errors) - 1)
    ax.plot(sorted_errors, yvals * 100, linewidth=2, color='darkblue')
    ax.axvline(5.0, color='orange', linestyle=':', linewidth=2, label='5mm threshold')
    ax.axhline(stats['LA_5mm'], color='orange', linestyle=':', linewidth=1,
               label=f'LA<5mm: {stats["LA_5mm"]:.1f}%')
    ax.set_xlabel('Error Threshold (mm)', fontsize=11)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    #3. Box Plot by Region
    ax = axes[1, 0]
    regions = {}
    for e in errors:
        r = e["region"]
        regions.setdefault(r, []).append(e["error_mm"])
    
    region_names = list(regions.keys())
    region_data = [regions[r] for r in region_names]
    
    bp = ax.boxplot(region_data, labels=region_names, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.axhline(5.0, color='orange', linestyle=':', linewidth=1.5, label='5mm threshold')
    ax.set_ylabel('Error (mm)', fontsize=11)
    ax.set_title('Error by Anatomical Region', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. Q-Q Plot (normality check)
    ax = axes[1, 1]
    from scipy import stats as scipy_stats
    scipy_stats.probplot(error_values, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "error_distributions.png")
    plt.savefig(save_path, dpi=200)
    print(f" Saved distribution plots: {save_path}")
    plt.close()


# REPORTING

def print_statistics_report(stats, regional_stats, icp_info=None):
    """Print comprehensive statistics report."""
    print("STATISTICAL ANALYSIS RESULTS")
    
    if icp_info:
        print(f"\nAlignment Method: ICP")
        print(f"  Iterations: {icp_info['iterations']}")
        print(f"  Final RMS Error: {icp_info['rms_error']:.2f} mm")
    
    print(f"\n{'OVERALL ACCURACY':^70}")
    print(f"  Sample Size (n):          {stats['n']}")
    print(f"  Mean Error:               {stats['mean']:.2f} ± {stats['std']:.2f} mm")
    print(f"  Median Error (MAD):       {stats['median']:.2f} ± {stats['mad']:.2f} mm")
    print(f"  Range:                    [{stats['min']:.2f}, {stats['max']:.2f}] mm")
    print(f"  90th Percentile:          {stats['p90']:.2f} mm")
    print(f"  95th Percentile:          {stats['p95']:.2f} mm")
    
    print(f"\n{'LOCALIZATION ACCURACY (Taberna 2019)':^70}")
    print(f"  LA < 5mm:                 {stats['LA_5mm']:.1f}% ({int(stats['LA_5mm']*stats['n']/100)}/{stats['n']} electrodes)")
    print(f"  LA < 10mm:                {stats['LA_10mm']:.1f}% ({int(stats['LA_10mm']*stats['n']/100)}/{stats['n']} electrodes)")
    
    print(f"\n{'REGIONAL BREAKDOWN':^70}")
    print(f"  {'Region':<15} {'n':<5} {'Mean (mm)':<12} {'Median (mm)':<12} {'SD (mm)':<10}")
    for region, rstats in sorted(regional_stats.items()):
        print(f"  {region:<15} {rstats['n']:<5} "
              f"{rstats['mean']:>6.2f}       "
              f"{rstats['median']:>6.2f}         "
              f"{rstats['std']:>6.2f}")
    


def save_results_csv(errors, stats, output_dir):
    """Save detailed results to CSV for further analysis."""
    import csv
    
    # Per-electrode results
    csv_path = os.path.join(output_dir, "electrode_errors.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'electrode_label', 'error_mm', 'region', 
            'position_x', 'position_y', 'position_z',
            'matched_scanner_label'
        ])
        writer.writeheader()
        for e in errors:
            writer.writerow({
                'electrode_label': e['label'],
                'error_mm': f"{e['error_mm']:.3f}",
                'region': e['region'],
                'position_x': f"{e['position'][0]:.2f}",
                'position_y': f"{e['position'][1]:.2f}",
                'position_z': f"{e['position'][2]:.2f}",
                'matched_scanner_label': e['matched_scanner_label']
            })
    
    print(f"Saved detailed results: {csv_path}")
    
    # Summary statistics
    summary_path = os.path.join(output_dir, "summary_statistics.txt")
    with open(summary_path, 'w') as f:
        f.write("STATISTICAL SUMMARY\n")
        f.write(f"Mean Error:     {stats['mean']:.2f} ± {stats['std']:.2f} mm\n")
        f.write(f"Median Error:   {stats['median']:.2f} mm (MAD: {stats['mad']:.2f})\n")
        f.write(f"Range:          [{stats['min']:.2f}, {stats['max']:.2f}] mm\n")
        f.write(f"LA < 5mm:       {stats['LA_5mm']:.1f}%\n")
        f.write(f"LA < 10mm:      {stats['LA_10mm']:.1f}%\n")
        f.write(f"Sample Size:    {stats['n']}\n")
    
    print(f"Saved summary: {summary_path}")


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Statistical validation of EEG electrode localization pipeline"
    )
    parser.add_argument("-p", "--pipeline", required=True,
                       help="Pipeline JSON output")
    parser.add_argument("-g", "--ground_truth", required=True,
                       help="3D Scanner ELC file")
    parser.add_argument("-o", "--output", default="validation_results",
                       help="Output directory")
    parser.add_argument("--use-icp", action="store_true",
                       help="Use ICP alignment (default: landmark-based)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("EEG ELECTRODE LOCALIZATION - STATISTICAL VALIDATION")
    print(f"\nPipeline Output:  {args.pipeline}")
    print(f"Ground Truth:     {args.ground_truth}")
    print(f"Output Directory: {args.output}")
    
    # ========== LOAD DATA ==========
    print("\n[1/5] Loading data...")
    p_data = parse_pipeline_json(args.pipeline)
    s_data = parse_scanner_elc(args.ground_truth)
    
    if not p_data or not s_data:
        print("  ERROR: Failed to load data")
        return
    
    print(f"Pipeline: {len(p_data)} points")
    print(f"Scanner:  {len(s_data)} points")
    
    # ========== ALIGNMENT ==========
    print("\n[2/5] Aligning point clouds...")
    
    icp_info = None
    if args.use_icp:
        print("  Using ICP alignment (Clausner 2017, Taberna 2019)...")
        
        # Get electrode coordinates (exclude landmarks)
        landmarks = ["NAS", "LPA", "RPA", "INION"]
        p_electrodes = np.array([v for k,v in p_data.items() if k not in landmarks])
        s_electrodes = np.array([v for k,v in s_data.items() if k not in landmarks])
        
        if len(p_electrodes) < 3 or len(s_electrodes) < 3:
            print("  ERROR: Not enough electrodes for ICP")
            return
        
        # Run ICP
        aligned_coords, transform, rms_error = icp_alignment(
            p_electrodes, s_electrodes, max_iterations=50
        )
        
        # Apply transformation to ALL points (including landmarks)
        p_aligned = {}
        all_coords = np.array([v for v in p_data.values()])
        all_labels = list(p_data.keys())
        
        transformed = transform['scale'] * (all_coords - all_coords.mean(axis=0)) @ transform['rotation'].T + transform['translation'] + all_coords.mean(axis=0)
        
        for i, label in enumerate(all_labels):
            p_aligned[label] = transformed[i]
        
        icp_info = {
            'iterations': transform['iterations'],
            'rms_error': rms_error,
            'scale': transform['scale']
        }
        
        print(f"ICP converged in {icp_info['iterations']} iterations")
        print(f"RMS error: {icp_info['rms_error']:.2f} mm")
        
    else:
        print("  Using landmark-based alignment...")
        p_aligned, transform = landmark_based_alignment(p_data, s_data)
        
        if not p_aligned:
            print("  ERROR: Alignment failed (missing landmarks)")
            return
        
        print(f"Aligned using NAS, LPA, RPA")
    
    # ========== COMPUTE ERRORS ==========
    print("\n[3/5] Computing nearest-neighbor errors...")
    errors = compute_nearest_neighbor_errors(p_aligned, s_data)
    
    if not errors:
        print("  ERROR: No electrodes to compare")
        return
    
    print(f"Computed errors for {len(errors)} electrodes")
    
    # ========== STATISTICS ==========
    print("\n[4/5] Computing statistics...")
    stats = compute_statistics(errors)
    regional_stats = regional_statistics(errors)
    
    print_statistics_report(stats, regional_stats, icp_info)
    
    # ========== VISUALIZATION ==========
    print("\n[5/5] Generating visualizations...")
    plot_discrete_electrode_errors(errors, args.output)
    plot_error_distributions(errors, stats, args.output)
    
    # ========== SAVE RESULTS ==========
    save_results_csv(errors, stats, args.output)
    
    print("VALIDATION COMPLETE")
    print(f"\nResults saved to: {args.output}")
    print("\nFiles generated:")
    print(f"  - discrete_electrode_errors.png  (No interpolation!)")
    print(f"  - error_distributions.png")
    print(f"  - electrode_errors.csv")
    print(f"  - summary_statistics.txt")
    print("\n")


if __name__ == "__main__":
    main()