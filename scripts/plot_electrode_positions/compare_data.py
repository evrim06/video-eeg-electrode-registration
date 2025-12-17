import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# --- PATHS ---
# Make sure these match your actual folder structure
BASE_DIR = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\results"
RAW_PATH = os.path.join(BASE_DIR, "tracking_raw.pkl")
SMOOTH_PATH = os.path.join(BASE_DIR, "tracking_smoothed.pkl")

def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Load both files
raw_data = load_data(RAW_PATH)
smooth_data = load_data(SMOOTH_PATH)

# Get list of all electrode IDs (excluding landmarks 0, 1, 2)
all_ids = set()
for frame in raw_data.values():
    for obj_id in frame.keys():
        if obj_id >= 3:
            all_ids.add(obj_id)
all_ids = sorted(list(all_ids))

print(f"Found {len(all_ids)} electrodes.")

# --- CONFIGURATION ---
# How many electrodes to compare?
NUM_TO_PLOT = 3 
# Select random electrodes to inspect
selected_ids = random.sample(all_ids, min(len(all_ids), NUM_TO_PLOT))

# Create Plot
fig, axes = plt.subplots(1, len(selected_ids), figsize=(15, 5))
if len(selected_ids) == 1: axes = [axes]

for ax, obj_id in zip(axes, selected_ids):
    # Extract trajectories
    raw_x, raw_y = [], []
    smooth_x, smooth_y = [], []
    frames = []
    
    # Sort frames to ensure correct order
    sorted_frames = sorted(raw_data.keys())
    
    for f_idx in sorted_frames:
        # Raw Data
        if obj_id in raw_data[f_idx]:
            raw_x.append(raw_data[f_idx][obj_id][0])
            raw_y.append(raw_data[f_idx][obj_id][1])
        else:
            raw_x.append(None)
            raw_y.append(None)
            
        # Smoothed Data
        if f_idx in smooth_data and obj_id in smooth_data[f_idx]:
            smooth_x.append(smooth_data[f_idx][obj_id][0])
            smooth_y.append(smooth_data[f_idx][obj_id][1])
        else:
            smooth_x.append(None)
            smooth_y.append(None)
            
        frames.append(f_idx)

    # Plot Raw (High Jitter)
    ax.plot(raw_x, raw_y, 'r.', markersize=2, alpha=0.3, label='Raw (Noisy)')
    ax.plot(raw_x, raw_y, 'r-', linewidth=0.5, alpha=0.3)
    
    # Plot Smoothed (Clean)
    ax.plot(smooth_x, smooth_y, 'b-', linewidth=2, label='Smoothed')
    
    ax.set_title(f"Electrode ID {obj_id}")
    ax.invert_yaxis() # Match image coordinates
    ax.axis('equal')
    ax.legend()

plt.suptitle("Comparison: Raw vs. Smoothed Trajectories", fontsize=16)
plt.tight_layout()
plt.show()