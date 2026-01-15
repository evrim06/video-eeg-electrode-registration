"""
================================================================================
SCRIPT 3: COMPREHENSIVE ACCURACY ANALYSIS
================================================================================

Compares video-based pipeline output to 3D scanner ground truth following
established methodology from:
    - Taberna et al. (2019)
    - Clausner et al. (2017)
    - Mazzonetto et al. (2022)

Analysis Steps:
    1. ICP Alignment - Align coordinate systems (Clausner 2017, Taberna 2019)
    2. Euclidean Distance Error - Per-electrode error measurement
    3. Localization Accuracy (LA<5mm) - Percentage within threshold (Taberna 2019)
    4. Descriptive Statistics - Mean, Median, SD, MAD (Mazzonetto 2022)
    5. Wilcoxon Signed-Rank Test - Statistical comparison (all papers)

Usage:
    # Single recording comparison:
    python script3.py -p results/electrodes_3d.json -g data/scanner/recording.elc
    
    # Batch analysis (multiple recordings):
    python script3.py --batch -p results/ -g data/scanner/ -o analysis_results/

================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon, shapiro, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

print("=" * 70)
print("SCRIPT 3: COMPREHENSIVE ACCURACY ANALYSIS")
print("Following Taberna (2019), Clausner (2017), Mazzonetto (2022)")
print("=" * 70)

# Accuracy thresholds from literature
THRESHOLD_EXCELLENT = 3.0   # mm - excellent localization
THRESHOLD_GOOD = 5.0        # mm - Taberna et al. threshold for minimal EEG source effect
THRESHOLD_ACCEPTABLE = 10.0 # mm - acceptable for most applications

# Fiducial names (various conventions)
FIDUCIAL_VARIANTS = {
    "NAS": ["NAS", "NASION", "NA", "NZ"],
    "LPA": ["LPA", "LEFT", "LHJ", "L", "A1"],
    "RPA": ["RPA", "RIGHT", "RHJ", "R", "A2"],
    "INION": ["INION", "INI", "IZ"]
}


# ==============================================================================
# FILE PARSING
# ==============================================================================

def parse_elc_file(filepath):
    """
    Parse .elc electrode position file.
    Returns dict: {label: np.array([x, y, z])}
    """
    positions = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    in_positions = False
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
        
        if line.lower().startswith('positions'):
            in_positions = True
            continue
        
        if line.lower().startswith('labels'):
            in_positions = False
            continue
        
        if in_positions:
            parts = line.split()
            if len(parts) >= 4:
                label = parts[0].upper()
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    positions[label] = np.array([x, y, z])
                except ValueError:
                    continue
    
    return positions


def parse_pipeline_json(filepath):
    """Parse pipeline JSON output."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    positions = {}
    
    # Parse landmarks
    for name, value in data.get("landmarks", {}).items():
        if isinstance(value, dict):
            pos = value.get("position", value)
        else:
            pos = value
        if pos is not None:
            positions[name.upper()] = np.array(pos)
    
    # Parse electrodes
    for name, value in data.get("electrodes", {}).items():
        if isinstance(value, dict):
            pos = value.get("position", value)
        else:
            pos = value
        if pos is not None:
            positions[name.upper()] = np.array(pos)
    
    return positions, data


def normalize_electrode_name(name):
    """Normalize electrode name for matching."""
    name = name.upper().strip()
    
    # Check if it's a fiducial
    for standard, variants in FIDUCIAL_VARIANTS.items():
        if name in variants:
            return standard
    
    return name


def separate_landmarks_electrodes(positions):
    """Separate landmarks from electrodes."""
    landmarks = {}
    electrodes = {}
    
    all_fiducials = set()
    for variants in FIDUCIAL_VARIANTS.values():
        all_fiducials.update(variants)
    
    for name, pos in positions.items():
        normalized = normalize_electrode_name(name)
        if normalized in FIDUCIAL_VARIANTS or name in all_fiducials:
            landmarks[normalized] = pos
        else:
            electrodes[name] = pos
    
    return landmarks, electrodes


# ==============================================================================
# ICP ALIGNMENT (Iterative Closest Point)
# ==============================================================================

def icp_align(source_points, target_points, max_iterations=100, tolerance=1e-6):
    """
    Iterative Closest Point alignment.
    Aligns source points to target points.
    
    Based on Clausner et al. (2017) and Taberna et al. (2019) methodology.
    
    Args:
        source_points: Nx3 array of source coordinates
        target_points: Mx3 array of target coordinates
        max_iterations: Maximum ICP iterations
        tolerance: Convergence tolerance
    
    Returns:
        rotation: 3x3 rotation matrix
        translation: 3x1 translation vector
        scale: scalar scale factor
        transformed_source: Aligned source points
        final_error: Mean squared error after alignment
    """
    source = source_points.copy()
    
    prev_error = np.inf
    
    # Initialize transformation
    R = np.eye(3)
    t = np.zeros(3)
    s = 1.0
    
    for iteration in range(max_iterations):
        # Step 1: Find closest points in target for each source point
        distances = cdist(source, target_points)
        closest_indices = np.argmin(distances, axis=1)
        closest_points = target_points[closest_indices]
        
        # Step 2: Compute transformation using SVD (Procrustes)
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(closest_points, axis=0)
        
        source_centered = source - source_centroid
        target_centered = closest_points - target_centroid
        
        # Compute scale
        source_scale = np.sqrt(np.sum(source_centered ** 2))
        target_scale = np.sqrt(np.sum(target_centered ** 2))
        
        if source_scale > 1e-10 and target_scale > 1e-10:
            s_new = target_scale / source_scale
        else:
            s_new = 1.0
        
        # Compute rotation
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R_new = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T
        
        # Compute translation
        t_new = target_centroid - s_new * (R_new @ source_centroid)
        
        # Apply transformation
        source = s_new * (source_points @ R_new.T) + t_new
        
        # Update cumulative transformation
        R = R_new @ R
        t = s_new * (R_new @ t) + t_new
        s = s_new * s
        
        # Compute error
        error = np.mean(np.min(distances, axis=1) ** 2)
        
        # Check convergence
        if abs(prev_error - error) < tolerance:
            break
        
        prev_error = error
    
    return R, t, s, source, np.sqrt(error)


def landmark_alignment(source_landmarks, target_landmarks):
    """
    Align using landmarks (fiducials) as reference points.
    Falls back to this if ICP fails or for initial alignment.
    """
    # Find common landmarks
    common = []
    for key in source_landmarks:
        if key in target_landmarks:
            common.append(key)
    
    if len(common) < 3:
        return None
    
    source_pts = np.array([source_landmarks[k] for k in common])
    target_pts = np.array([target_landmarks[k] for k in common])
    
    # Procrustes alignment
    source_centroid = np.mean(source_pts, axis=0)
    target_centroid = np.mean(target_pts, axis=0)
    
    source_centered = source_pts - source_centroid
    target_centered = target_pts - target_centroid
    
    source_scale = np.sqrt(np.sum(source_centered ** 2))
    target_scale = np.sqrt(np.sum(target_centered ** 2))
    
    s = target_scale / source_scale if source_scale > 1e-10 else 1.0
    
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = target_centroid - s * (R @ source_centroid)
    
    return {"rotation": R, "translation": t, "scale": s, "landmarks_used": common}


def apply_transform(points_dict, R, t, s):
    """Apply transformation to all points in a dictionary."""
    transformed = {}
    for name, pos in points_dict.items():
        transformed[name] = s * (R @ pos) + t
    return transformed


# ==============================================================================
# ELECTRODE MATCHING
# ==============================================================================

def match_electrodes_hungarian(pipeline_positions, gt_positions, max_distance=20.0):
    """
    Match pipeline electrodes to ground truth using Hungarian algorithm.
    Optimal bipartite matching that minimizes total distance.
    """
    p_names = list(pipeline_positions.keys())
    gt_names = list(gt_positions.keys())
    
    if not p_names or not gt_names:
        return [], p_names, gt_names
    
    # Compute distance matrix
    p_positions = np.array([pipeline_positions[n] for n in p_names])
    gt_positions_arr = np.array([gt_positions[n] for n in gt_names])
    
    dist_matrix = cdist(p_positions, gt_positions_arr)
    
    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    matches = []
    matched_p = set()
    matched_gt = set()
    
    for i, j in zip(row_ind, col_ind):
        dist = dist_matrix[i, j]
        if dist <= max_distance:
            matches.append({
                "pipeline_name": p_names[i],
                "gt_name": gt_names[j],
                "distance_mm": float(dist),
                "pipeline_pos": p_positions[i].tolist(),
                "gt_pos": gt_positions_arr[j].tolist()
            })
            matched_p.add(p_names[i])
            matched_gt.add(gt_names[j])
    
    unmatched_p = [n for n in p_names if n not in matched_p]
    unmatched_gt = [n for n in gt_names if n not in matched_gt]
    
    return matches, unmatched_p, unmatched_gt


# ==============================================================================
# ACCURACY METRICS (Following Literature)
# ==============================================================================

def compute_euclidean_errors(matches):
    """
    Compute Euclidean distance error for each matched electrode.
    This is the primary metric used in all referenced papers.
    """
    errors = [m["distance_mm"] for m in matches]
    return np.array(errors)


def compute_localization_accuracy(errors, threshold=5.0):
    """
    Compute Localization Accuracy: percentage of electrodes within threshold.
    
    Following Taberna et al. (2019):
    "5 mm threshold because it's known to have minimal effect on EEG source localization"
    
    Args:
        errors: Array of Euclidean distance errors in mm
        threshold: Distance threshold in mm (default 5.0)
    
    Returns:
        percentage: Percentage of electrodes within threshold
        count: Number of electrodes within threshold
        total: Total number of electrodes
    """
    if len(errors) == 0:
        return 0.0, 0, 0
    
    within_threshold = np.sum(errors <= threshold)
    percentage = 100.0 * within_threshold / len(errors)
    
    return percentage, int(within_threshold), len(errors)


def compute_descriptive_statistics(errors):
    """
    Compute descriptive statistics following Mazzonetto et al. (2022).
    
    Primary metrics (robust to outliers):
        - Median
        - MAD (Mean Absolute Deviation)
    
    Secondary metrics (for comparison across studies):
        - Mean
        - SD (Standard Deviation)
        - Min, Max
        - Percentiles (25th, 75th, 95th)
    """
    if len(errors) == 0:
        return None
    
    stats = {
        # Primary metrics (Mazzonetto 2022)
        "median_mm": float(np.median(errors)),
        "mad_mm": float(np.mean(np.abs(errors - np.mean(errors)))),  # Mean Absolute Deviation
        
        # Secondary metrics (Clausner 2017, Taberna 2019)
        "mean_mm": float(np.mean(errors)),
        "std_mm": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,  # Sample std
        
        # Range
        "min_mm": float(np.min(errors)),
        "max_mm": float(np.max(errors)),
        
        # Percentiles
        "percentile_25_mm": float(np.percentile(errors, 25)),
        "percentile_75_mm": float(np.percentile(errors, 75)),
        "percentile_95_mm": float(np.percentile(errors, 95)),
        
        # IQR (Interquartile Range)
        "iqr_mm": float(np.percentile(errors, 75) - np.percentile(errors, 25)),
        
        # Sample size
        "n_electrodes": len(errors)
    }
    
    return stats


def compute_localization_accuracy_tiers(errors):
    """
    Compute localization accuracy at multiple thresholds.
    """
    tiers = {}
    
    for threshold, name in [(3.0, "excellent_3mm"), 
                            (5.0, "good_5mm"), 
                            (10.0, "acceptable_10mm"),
                            (15.0, "within_15mm")]:
        pct, count, total = compute_localization_accuracy(errors, threshold)
        tiers[name] = {
            "percentage": pct,
            "count": count,
            "total": total,
            "threshold_mm": threshold
        }
    
    return tiers


# ==============================================================================
# STATISTICAL TESTS
# ==============================================================================

def wilcoxon_signed_rank_test(errors1, errors2):
    """
    Wilcoxon Signed-Rank Test for paired samples.
    
    Used by Clausner (2017), Taberna (2019), Mazzonetto (2022)
    to compare errors between two methods.
    
    H0: The median difference between paired observations is zero.
    
    Args:
        errors1: Error array from method 1 (e.g., pipeline)
        errors2: Error array from method 2 (e.g., another method)
    
    Returns:
        statistic: Test statistic
        p_value: Two-sided p-value
        interpretation: String interpretation
    """
    if len(errors1) != len(errors2):
        return None, None, "Arrays must have same length for paired test"
    
    if len(errors1) < 5:
        return None, None, "Need at least 5 samples for Wilcoxon test"
    
    try:
        stat, p = wilcoxon(errors1, errors2)
        
        if p < 0.001:
            interp = "Highly significant difference (p < 0.001)"
        elif p < 0.01:
            interp = "Very significant difference (p < 0.01)"
        elif p < 0.05:
            interp = "Significant difference (p < 0.05)"
        else:
            interp = "No significant difference (p >= 0.05)"
        
        return float(stat), float(p), interp
    
    except Exception as e:
        return None, None, f"Test failed: {str(e)}"


def normality_test(errors):
    """
    Shapiro-Wilk test for normality.
    Helps determine if parametric or non-parametric tests are appropriate.
    """
    if len(errors) < 3:
        return None, None, None
    
    try:
        stat, p = shapiro(errors)
        is_normal = p > 0.05
        return float(stat), float(p), is_normal
    except:
        return None, None, None


# ==============================================================================
# BATCH ANALYSIS
# ==============================================================================

def find_matching_files(pipeline_dir, gt_dir):
    """
    Find matching pipeline output and ground truth files.
    Matches by filename similarity.
    """
    pipeline_files = []
    gt_files = []
    
    # Find pipeline files
    for f in os.listdir(pipeline_dir):
        if f.endswith('.json') and ('electrode' in f.lower() or '3d' in f.lower()):
            pipeline_files.append(os.path.join(pipeline_dir, f))
    
    # Find ground truth files
    for f in os.listdir(gt_dir):
        if f.endswith('.elc'):
            gt_files.append(os.path.join(gt_dir, f))
    
    # Sort both lists
    pipeline_files.sort()
    gt_files.sort()
    
    # If same number of files, assume they correspond 1:1
    if len(pipeline_files) == len(gt_files):
        return list(zip(pipeline_files, gt_files))
    
    # Otherwise, try to match by name
    matches = []
    for p_path in pipeline_files:
        p_base = os.path.splitext(os.path.basename(p_path))[0].lower()
        p_base = p_base.replace('electrodes_3d', '').replace('electrodes', '').strip('_-')
        
        for g_path in gt_files:
            g_base = os.path.splitext(os.path.basename(g_path))[0].lower()
            
            if p_base in g_base or g_base in p_base:
                matches.append((p_path, g_path))
                break
    
    return matches


def analyze_single_recording(pipeline_path, gt_path, max_distance=20.0, verbose=True):
    """
    Analyze a single recording (pipeline vs ground truth).
    Returns comprehensive analysis results.
    """
    results = {
        "pipeline_file": pipeline_path,
        "gt_file": gt_path,
        "success": False
    }
    
    # Load files
    try:
        pipeline_positions, pipeline_data = parse_pipeline_json(pipeline_path)
        gt_positions = parse_elc_file(gt_path)
    except Exception as e:
        results["error"] = f"Failed to load files: {str(e)}"
        return results
    
    if verbose:
        print(f"\n  Pipeline: {len(pipeline_positions)} positions")
        print(f"  Ground truth: {len(gt_positions)} positions")
    
    # Separate landmarks and electrodes
    p_landmarks, p_electrodes = separate_landmarks_electrodes(pipeline_positions)
    gt_landmarks, gt_electrodes = separate_landmarks_electrodes(gt_positions)
    
    if verbose:
        print(f"  Pipeline: {len(p_landmarks)} landmarks, {len(p_electrodes)} electrodes")
        print(f"  Ground truth: {len(gt_landmarks)} landmarks, {len(gt_electrodes)} electrodes")
    
    # Step 1: Alignment
    if verbose:
        print("\n  --- Alignment ---")
    
    # Try landmark-based alignment first
    landmark_transform = landmark_alignment(p_landmarks, gt_landmarks)
    
    if landmark_transform is not None:
        if verbose:
            print(f"  Landmark alignment using: {landmark_transform['landmarks_used']}")
            print(f"  Scale factor: {landmark_transform['scale']:.4f}")
        
        # Apply transformation
        aligned_positions = apply_transform(
            pipeline_positions,
            landmark_transform["rotation"],
            landmark_transform["translation"],
            landmark_transform["scale"]
        )
        
        # Refine with ICP on electrodes
        aligned_landmarks, aligned_electrodes = separate_landmarks_electrodes(aligned_positions)
        
        if len(aligned_electrodes) >= 4 and len(gt_electrodes) >= 4:
            source_pts = np.array(list(aligned_electrodes.values()))
            target_pts = np.array(list(gt_electrodes.values()))
            
            R_icp, t_icp, s_icp, _, icp_error = icp_align(source_pts, target_pts)
            
            if verbose:
                print(f"  ICP refinement error: {icp_error:.2f} mm")
            
            # Apply ICP refinement
            aligned_positions = apply_transform(aligned_positions, R_icp, t_icp, s_icp)
        
        results["alignment"] = {
            "method": "landmark + ICP",
            "landmarks_used": landmark_transform["landmarks_used"],
            "scale_factor": float(landmark_transform["scale"])
        }
    else:
        if verbose:
            print("  WARNING: Landmark alignment failed, using ICP only")
        
        # ICP only
        source_pts = np.array(list(pipeline_positions.values()))
        target_pts = np.array(list(gt_positions.values()))
        
        R_icp, t_icp, s_icp, _, icp_error = icp_align(source_pts, target_pts)
        aligned_positions = apply_transform(pipeline_positions, R_icp, t_icp, s_icp)
        
        results["alignment"] = {
            "method": "ICP only",
            "icp_error_mm": float(icp_error)
        }
    
    # Separate aligned positions
    aligned_landmarks, aligned_electrodes = separate_landmarks_electrodes(aligned_positions)
    
    # Step 2: Match electrodes
    if verbose:
        print("\n  --- Electrode Matching ---")
    
    matches, unmatched_p, unmatched_gt = match_electrodes_hungarian(
        aligned_electrodes, gt_electrodes, max_distance
    )
    
    if verbose:
        print(f"  Matched: {len(matches)}")
        print(f"  Unmatched (pipeline): {len(unmatched_p)}")
        print(f"  Unmatched (ground truth): {len(unmatched_gt)}")
    
    results["matching"] = {
        "n_matched": len(matches),
        "n_unmatched_pipeline": len(unmatched_p),
        "n_unmatched_gt": len(unmatched_gt),
        "unmatched_pipeline": unmatched_p,
        "unmatched_gt": unmatched_gt,
        "electrode_mapping": {m["pipeline_name"]: m["gt_name"] for m in matches}
    }
    
    # Step 3: Compute errors
    errors = compute_euclidean_errors(matches)
    
    if len(errors) == 0:
        results["error"] = "No electrodes matched"
        return results
    
    # Step 4: Descriptive statistics
    stats = compute_descriptive_statistics(errors)
    results["statistics"] = stats
    
    # Step 5: Localization accuracy
    la_tiers = compute_localization_accuracy_tiers(errors)
    results["localization_accuracy"] = la_tiers
    
    # Step 6: Normality test
    norm_stat, norm_p, is_normal = normality_test(errors)
    results["normality_test"] = {
        "shapiro_statistic": norm_stat,
        "p_value": norm_p,
        "is_normal": is_normal
    }
    
    # Per-electrode errors
    results["per_electrode_errors"] = matches
    
    # Landmark errors (if aligned)
    landmark_errors = []
    for lm_name in aligned_landmarks:
        if lm_name in gt_landmarks:
            error = np.linalg.norm(aligned_landmarks[lm_name] - gt_landmarks[lm_name])
            landmark_errors.append({
                "landmark": lm_name,
                "error_mm": float(error)
            })
    results["landmark_errors"] = landmark_errors
    
    results["success"] = True
    
    # Print summary
    if verbose:
        print("\n  --- Results Summary ---")
        print(f"  Mean error: {stats['mean_mm']:.2f} ± {stats['std_mm']:.2f} mm")
        print(f"  Median error: {stats['median_mm']:.2f} mm (MAD: {stats['mad_mm']:.2f})")
        print(f"  Range: {stats['min_mm']:.2f} - {stats['max_mm']:.2f} mm")
        print(f"\n  Localization Accuracy:")
        print(f"    < 3mm (excellent): {la_tiers['excellent_3mm']['percentage']:.1f}%")
        print(f"    < 5mm (good):      {la_tiers['good_5mm']['percentage']:.1f}%")
        print(f"    < 10mm (acceptable): {la_tiers['acceptable_10mm']['percentage']:.1f}%")
    
    return results


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_error_histogram(errors, output_path, title="Electrode Localization Errors"):
    """Create histogram of electrode errors."""
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available for visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    
    # Color code by threshold
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        right_edge = bins[i + 1]
        center = (left_edge + right_edge) / 2
        if center <= 3:
            patch.set_facecolor('green')
        elif center <= 5:
            patch.set_facecolor('lightgreen')
        elif center <= 10:
            patch.set_facecolor('yellow')
        else:
            patch.set_facecolor('red')
    
    # Add threshold lines
    ax.axvline(x=3, color='green', linestyle='--', label='3mm (excellent)')
    ax.axvline(x=5, color='orange', linestyle='--', label='5mm (good)')
    ax.axvline(x=10, color='red', linestyle='--', label='10mm (acceptable)')
    
    # Add statistics
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    ax.axvline(x=mean_err, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean_err:.1f}mm')
    ax.axvline(x=median_err, color='purple', linestyle='-', linewidth=2, label=f'Median: {median_err:.1f}mm')
    
    ax.set_xlabel('Localization Error (mm)')
    ax.set_ylabel('Number of Electrodes')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved histogram: {output_path}")


def create_error_boxplot(all_errors, labels, output_path, title="Error Comparison"):
    """Create boxplot comparing errors across recordings."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(all_errors, labels=labels, patch_artist=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    # Add threshold lines
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5mm threshold')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10mm threshold')
    
    ax.set_xlabel('Recording')
    ax.set_ylabel('Localization Error (mm)')
    ax.set_title(title)
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_summary_table(all_results, output_path):
    """Create summary table of all recordings."""
    if not HAS_PANDAS:
        # Create simple text table
        with open(output_path.replace('.csv', '.txt'), 'w') as f:
            f.write("Recording\tN_Electrodes\tMean(mm)\tSD(mm)\tMedian(mm)\tLA<5mm(%)\n")
            for r in all_results:
                if r["success"]:
                    name = os.path.basename(r["pipeline_file"])
                    stats = r["statistics"]
                    la = r["localization_accuracy"]["good_5mm"]["percentage"]
                    f.write(f"{name}\t{stats['n_electrodes']}\t{stats['mean_mm']:.2f}\t"
                           f"{stats['std_mm']:.2f}\t{stats['median_mm']:.2f}\t{la:.1f}\n")
        return
    
    rows = []
    for r in all_results:
        if r["success"]:
            stats = r["statistics"]
            la = r["localization_accuracy"]
            rows.append({
                "Recording": os.path.basename(r["pipeline_file"]),
                "N_Electrodes": stats["n_electrodes"],
                "Mean (mm)": round(stats["mean_mm"], 2),
                "SD (mm)": round(stats["std_mm"], 2),
                "Median (mm)": round(stats["median_mm"], 2),
                "MAD (mm)": round(stats["mad_mm"], 2),
                "Min (mm)": round(stats["min_mm"], 2),
                "Max (mm)": round(stats["max_mm"], 2),
                "LA<3mm (%)": round(la["excellent_3mm"]["percentage"], 1),
                "LA<5mm (%)": round(la["good_5mm"]["percentage"], 1),
                "LA<10mm (%)": round(la["acceptable_10mm"]["percentage"], 1),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved summary table: {output_path}")
    
    return df


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive accuracy analysis: Pipeline vs Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single recording:
  python script3.py -p results/electrodes_3d.json -g data/scanner/recording.elc
  
  # Batch analysis:
  python script3.py --batch -p results/ -g data/scanner/ -o analysis/
  
  # With visualization:
  python script3.py -p results/electrodes_3d.json -g data/scanner/recording.elc --plot
        """
    )
    
    parser.add_argument("-p", "--pipeline", required=True, 
                        help="Pipeline output JSON file or directory (for batch)")
    parser.add_argument("-g", "--ground_truth", required=True,
                        help="Ground truth .elc file or directory (for batch)")
    parser.add_argument("-o", "--output", 
                        help="Output file/directory for results")
    parser.add_argument("-d", "--max_distance", type=float, default=20.0,
                        help="Maximum matching distance in mm (default: 20)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: process multiple recordings")
    parser.add_argument("--plot", action="store_true",
                        help="Generate visualization plots")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch analysis
        print("\n--- Batch Analysis Mode ---")
        
        if not os.path.isdir(args.pipeline) or not os.path.isdir(args.ground_truth):
            print("ERROR: For batch mode, -p and -g must be directories")
            sys.exit(1)
        
        output_dir = args.output or "analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Find matching files
        file_pairs = find_matching_files(args.pipeline, args.ground_truth)
        print(f"  Found {len(file_pairs)} matching file pairs")
        
        if len(file_pairs) == 0:
            print("  ERROR: No matching files found")
            print(f"  Pipeline dir: {args.pipeline}")
            print(f"  Ground truth dir: {args.ground_truth}")
            sys.exit(1)
        
        all_results = []
        all_errors = []
        labels = []
        
        for pipeline_path, gt_path in file_pairs:
            print(f"\n{'='*60}")
            print(f"Analyzing: {os.path.basename(pipeline_path)}")
            print(f"      vs:  {os.path.basename(gt_path)}")
            
            results = analyze_single_recording(pipeline_path, gt_path, args.max_distance)
            all_results.append(results)
            
            if results["success"]:
                errors = compute_euclidean_errors(results["per_electrode_errors"])
                all_errors.append(errors)
                labels.append(os.path.basename(pipeline_path)[:20])
        
        # Save individual results
        for i, results in enumerate(all_results):
            result_file = os.path.join(output_dir, f"analysis_{i+1}.json")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Create summary
        print("\n" + "=" * 70)
        print("BATCH ANALYSIS SUMMARY")
        print("=" * 70)
        
        successful = [r for r in all_results if r["success"]]
        print(f"\n  Successfully analyzed: {len(successful)}/{len(all_results)}")
        
        if successful:
            # Aggregate statistics
            all_means = [r["statistics"]["mean_mm"] for r in successful]
            all_medians = [r["statistics"]["median_mm"] for r in successful]
            all_la5 = [r["localization_accuracy"]["good_5mm"]["percentage"] for r in successful]
            
            print(f"\n  Overall Performance:")
            print(f"    Mean error (across recordings): {np.mean(all_means):.2f} ± {np.std(all_means):.2f} mm")
            print(f"    Median error (across recordings): {np.mean(all_medians):.2f} mm")
            print(f"    Average LA<5mm: {np.mean(all_la5):.1f}%")
            
            # Create summary table
            summary_path = os.path.join(output_dir, "summary.csv")
            create_summary_table(all_results, summary_path)
            
            # Create visualizations
            if args.plot and all_errors:
                hist_path = os.path.join(output_dir, "error_histogram_all.png")
                all_errors_flat = np.concatenate(all_errors)
                create_error_histogram(all_errors_flat, hist_path, "All Recordings - Error Distribution")
                
                if len(all_errors) > 1:
                    box_path = os.path.join(output_dir, "error_boxplot.png")
                    create_error_boxplot(all_errors, labels, box_path, "Error Comparison Across Recordings")
        
        # Save combined results
        combined_path = os.path.join(output_dir, "all_results.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_dir}/")
    
    else:
        # Single file analysis
        print("\n--- Single Recording Analysis ---")
        
        results = analyze_single_recording(args.pipeline, args.ground_truth, args.max_distance)
        
        if results["success"]:
            # Save results
            output_path = args.output or args.pipeline.replace('.json', '_analysis.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  Results saved to: {output_path}")
            
            # Create visualization
            if args.plot:
                errors = compute_euclidean_errors(results["per_electrode_errors"])
                plot_path = output_path.replace('.json', '_histogram.png')
                create_error_histogram(errors, plot_path)
            
            # Quality rating
            mean_err = results["statistics"]["mean_mm"]
            la5 = results["localization_accuracy"]["good_5mm"]["percentage"]
            
            print("\n" + "=" * 70)
            print("QUALITY ASSESSMENT")
            print("=" * 70)
            
            if mean_err < 3 and la5 > 90:
                quality = "EXCELLENT"
                symbol = "[***]"
            elif mean_err < 5 and la5 > 75:
                quality = "GOOD"
                symbol = "[**]"
            elif mean_err < 10 and la5 > 50:
                quality = "ACCEPTABLE"
                symbol = "[*]"
            else:
                quality = "NEEDS IMPROVEMENT"
                symbol = "[!]"
            
            print(f"\n  {symbol} Overall Quality: {quality}")
            print(f"  Mean error: {mean_err:.2f} mm")
            print(f"  Localization Accuracy (<5mm): {la5:.1f}%")
        
        else:
            print(f"\n  ERROR: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()