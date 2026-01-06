import numpy as np
import os

# Load reconstruction
recon = np.load("results/vggt_output/reconstruction.npz")

print("=== Script 2 Output Test ===\n")

# Check all required keys exist
required = ["depth", "intrinsics", "extrinsics", "frame_mapping_keys", "frame_mapping_values"]
for key in required:
    if key in recon:
        print(f"✓ {key}: {recon[key].shape}")
    else:
        print(f"✗ MISSING: {key}")

# Check frame mapping
keys = recon["frame_mapping_keys"]
vals = recon["frame_mapping_values"]
print(f"\nFrame mapping:")
print(f"  VGGT indices: {keys}")
print(f"  Script1 indices: {vals}")

# Check depth values
depth = np.squeeze(recon["depth"])
print(f"\nDepth stats:")
print(f"  Shape: {depth.shape}")
print(f"  Min: {depth.min():.4f}")
print(f"  Max: {depth.max():.4f}")
print(f"  Mean: {depth.mean():.4f}")

# Check extrinsics shape
ext = np.squeeze(recon["extrinsics"])
print(f"\nExtrinsics shape: {ext.shape}")
print(f"  Expected: (N, 4, 4) where N = {len(keys)}")

print("\n=== Test Complete ===")