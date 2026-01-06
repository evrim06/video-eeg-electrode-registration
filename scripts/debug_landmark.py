import json
import numpy as np

with open("results/electrodes_3d.json", "r") as f:
    data = json.load(f)

scale = data["measurement"]["scale_factor"]
print(f"Scale factor: {scale:.2f} mm/unit")

# Reverse calculate VGGT units
nas = np.array(data["landmarks"]["NAS"]) / scale
lpa = np.array(data["landmarks"]["LPA"]) / scale
rpa = np.array(data["landmarks"]["RPA"]) / scale

ear_dist_vggt = np.linalg.norm(rpa - lpa)
nas_to_center_vggt = np.linalg.norm(nas - (lpa + rpa) / 2)

print(f"\nIn VGGT units:")
print(f"  Ear-to-ear: {ear_dist_vggt:.4f}")
print(f"  NAS to center: {nas_to_center_vggt:.4f}")
print(f"  Ratio (NAS/ear): {nas_to_center_vggt / ear_dist_vggt:.2f}")
print(f"  Expected ratio: ~0.7-0.8")