import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_FILE = os.path.join(BASE_DIR, "results", "electrodes_3d.json")

def plot_3d_electrodes():
    if not os.path.exists(JSON_FILE):
        print(f"File not found: {JSON_FILE}")
        return

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    landmarks = data.get("landmarks", {})
    electrodes = data.get("electrodes", {})

    # Extract coordinates (with check for valid numbers)
    l_names, l_coords = [], []
    for name, pos in landmarks.items():
        # Skip notes or non-coordinate entries
        if not isinstance(pos, list) or len(pos) != 3:
            continue
        l_names.append(name)
        l_coords.append(pos)

    e_names, e_coords = [], []
    for name, pos in electrodes.items():
        # Skip notes or non-coordinate entries
        if not isinstance(pos, list) or len(pos) != 3:
            continue
        e_names.append(name)
        e_coords.append(pos)

    l_coords = np.array(l_coords)
    e_coords = np.array(e_coords)

    # Setup Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Electrodes (Blue Dots)
    if len(e_coords) > 0:
        ax.scatter(e_coords[:, 0], e_coords[:, 1], e_coords[:, 2], 
                   c='blue', marker='o', s=50, label='Electrodes')
        for i, txt in enumerate(e_names):
            ax.text(e_coords[i, 0], e_coords[i, 1], e_coords[i, 2], 
                    txt, size=8, zorder=1, color='k') 

    # Plot Landmarks (Red Squares)
    if len(l_coords) > 0:
        ax.scatter(l_coords[:, 0], l_coords[:, 1], l_coords[:, 2], 
                   c='red', marker='s', s=100, label='Landmarks')
        for i, txt in enumerate(l_names):
            ax.text(l_coords[i, 0], l_coords[i, 1], l_coords[i, 2], 
                    txt, size=10, zorder=10, weight='bold', color='red')

    # Draw Axes Lines for Reference
    ax.plot([-80, 80], [0, 0], [0, 0], 'k--', alpha=0.3, label="X Axis (LPA-RPA)")
    ax.plot([0, 0], [-80, 80], [0, 0], 'k-.', alpha=0.3, label="Y Axis (Inion-Nas)")
    ax.plot([0, 0], [0, 0], [-50, 100], 'k:', alpha=0.3, label="Z Axis (Vertical)")

    # Formatting
    ax.set_xlabel('X (Left - Right) [mm]')
    ax.set_ylabel('Y (Back - Front) [mm]')
    ax.set_zlabel('Z (Down - Up) [mm]')
    ax.set_title(f"Final 3D Reconstruction\nMeasurement Method: {data.get('measurement', {}).get('method', 'Unknown')}")
    ax.legend()
    
    # Equal Aspect Ratio Hack
    if len(l_coords) > 0 or len(e_coords) > 0:
        all_points = np.vstack([p for p in [l_coords, e_coords] if len(p) > 0])
        max_range = np.array([
            all_points[:,0].max()-all_points[:,0].min(), 
            all_points[:,1].max()-all_points[:,1].min(), 
            all_points[:,2].max()-all_points[:,2].min()
        ]).max() / 2.0
        
        mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == "__main__":
    plot_3d_electrodes()