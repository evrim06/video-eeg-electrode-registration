import json
import numpy as np

# Load results
with open("results/electrodes_3d.json", "r") as f:
    data = json.load(f)

print("=== Script 3 Output Test ===\n")

# Basic info
print(f"Landmarks: {list(data['landmarks'].keys())}")
print(f"Electrodes: {data['num_electrodes']}")
print(f"Measurement: {data['measurement']['method']} ({data['measurement']['ear_to_ear_mm']:.1f} mm)")

# Landmark positions
print("\nLandmark positions (mm):")
for name, pos in data["landmarks"].items():
    print(f"  {name}: ({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f})")

# Check distances
nas = np.array(data["landmarks"]["NAS"])
lpa = np.array(data["landmarks"]["LPA"])
rpa = np.array(data["landmarks"]["RPA"])
inion = np.array(data["landmarks"]["INION"])

ear_dist = np.linalg.norm(rpa - lpa)
nas_inion = np.linalg.norm(nas - inion)

print(f"\nDistances:")
print(f"  LPA-RPA: {ear_dist:.1f} mm (should be ~{data['measurement']['ear_to_ear_mm']:.1f})")
print(f"  NAS-INION: {nas_inion:.1f} mm")

# Check expected positions
print(f"\nExpected landmark positions:")
print(f"  LPA should be at ({-data['measurement']['ear_to_ear_mm']/2:.1f}, 0, 0)")
print(f"  RPA should be at ({data['measurement']['ear_to_ear_mm']/2:.1f}, 0, 0)")
print(f"  Actual LPA: ({lpa[0]:.1f}, {lpa[1]:.1f}, {lpa[2]:.1f})")
print(f"  Actual RPA: ({rpa[0]:.1f}, {rpa[1]:.1f}, {rpa[2]:.1f})")

# Alignment info
print(f"\nAlignment:")
print(f"  Method: {data['alignment']['method']}")
print(f"  Frames aligned: {data['alignment']['frames_aligned']}")
print(f"  Frames skipped: {data['alignment']['frames_skipped']}")

# Landmark observations
print(f"\nLandmark observations:")
for name, count in data["landmark_observations"].items():
    print(f"  {name}: {count} frames")

print("\n=== Test Complete ===")