from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

def get_small_masks_and_centroids(mask, min_size=50, max_size=3000):
    mask_uinit = mask.astype(np.uint8)
    labeled, num_features = label(mask_uinit)
    results = []
    for region_id in range(1, num_features + 1):
        region_mask = (labeled == region_id)
        size = region_mask.sum()
        if min_size <= size <= max_size:
            indices = np.argwhere(region_mask)
            centroid = indices.mean(axis=0)
            results.append((region_mask, tuple(centroid)))
    return results

# Load SAM2 with custom parameters
mask_generator  = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2-hiera-large",
    device="cpu",
    points_per_side=16,
    crop_n_layers=1,
    crop_overlap_ratio=512/1500,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.85,
    min_mask_region_area=0,
    box_nms_thresh=0.9,
    output_mode="binary_mask"
)

# Open video file
video_path = "data/IMG_2763.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
print("Video opened successfully.")

# Process first 5 frames
frame_count = 0

while frame_count < 5:
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read the frame {frame_count}.")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"\nProcessing frame {frame_count}...")
    masks = mask_generator.generate(frame_rgb)
    print("Number of masks:", len(masks))

    plt.figure(figsize=(8,6))
    plt.imshow(frame_rgb)
    plt.title(f"Frame {frame_count} with {len(masks)} masks")
    plt.axis('off')

    # For each mask, extract small regions and centroids, then plot
    for m in masks:
        binary_mask = m['segmentation']
        results = get_small_masks_and_centroids(binary_mask)
        for region_mask, (cx,cy) in results:
            plt.scatter(cx, cy, c='red', s=50)  # Plot centroid as a red dot

    plt.show()
    frame_count += 1

cap.release()  
print("Video processing completed.")