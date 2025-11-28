from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

# Helper 1: Split SAM2 mask into blobs, filter by size, compute centroid

def get_small_masks_and_centroids(mask, min_size=50, max_size=3000, min_circularity=0.50):
    """
    Given a SAM2 segmentation mask (boolean array):
    - split it into connected components (blobs)
    - filter blobs by area
    - compute circularity for each blob
    - keep only blobs that are round enough (like electrodes)
    - return masks + centroids
    """

    mask_uint = mask.astype(np.uint8)
    labeled, num_features = label(mask_uint)
    results = []

    for region_id in range(1, num_features + 1):

        region_mask = (labeled == region_id)
        size = region_mask.sum()

        # Filter by area
        if not (min_size <= size <= max_size):
            continue

        # Convert 0/1 mask → 0/255 grayscale for OpenCV
        region_uint8 = (region_mask.astype(np.uint8) * 255)

        # Find region contour for perimeter
        contours, _ = cv2.findContours(
            region_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter == 0:
            continue

        # Circularity = 4πA / P²
        circularity = 4 * np.pi * size / (perimeter ** 2)
        if circularity < min_circularity:
            continue   # Reject non-round shapes

        # Compute centroid
        coords = np.argwhere(region_mask)   # (y, x)
        cy, cx = coords.mean(axis=0)
        centroid = (int(cx), int(cy))       # return (x, y)

        results.append((region_mask, centroid, circularity))

    return results

#Find the cap mask

def get_head_mask(all_masks):
    """
    Select the LARGEST mask from SAM2 output.
    This usually corresponds to the head + EEG cap.
    """
    if len(all_masks) == 0:
        return None

    # Sort masks by largest area
    largest = max(all_masks, key=lambda x: x["area"])
    return largest["segmentation"].astype(bool)

# Load SAM2 model (change parameters for speed)

mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2-hiera-large",
    device="cpu",
    points_per_side=48,     # smaller = faster
    crop_n_layers=1,        # fewer layers = faster
    pred_iou_thresh=0.6,
    stability_score_thresh=0.85,
    min_mask_region_area=0,
    box_nms_thresh=0.9,
    output_mode="binary_mask"
)


# Load video

video_path = "data/IMG_2763.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(" Could not open video.")
    exit()

print(" Video opened successfully.")

# Process first N frames

N_FRAMES = 3
frame_index = 0

while frame_index < N_FRAMES:

    ret, frame = cap.read()
    if not ret:
        print(f" Could not read frame {frame_index}")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"\n➡ Processing frame {frame_index}...")

   
    # Resize frame for SAM2 (much faster), track original size

    original_h, original_w = frame_rgb.shape[:2]
    small_frame = cv2.resize(frame_rgb, (640, 360))  # you can adjust this

    scale_x = original_w / 640
    scale_y = original_h / 360

    # Run SAM2
    masks = mask_generator.generate(small_frame)
    print("SAM2 masks:", len(masks))

    electrode_centroids = []

    # 1. Split masks → blobs
    # 2. Filter components by size and circularity
    for m in masks:
        mask = m["segmentation"]   # boolean array
        blobs = get_small_masks_and_centroids(mask)

        for region_mask, (cx, cy), circ in blobs:

            # Scale centroid back to original frame resolution
            cx_scaled = int(cx * scale_x)
            cy_scaled = int(cy * scale_y)

            electrode_centroids.append((cx_scaled, cy_scaled))

    print("Detected electrode-like (round) blobs:", len(electrode_centroids))


    # Visualization
   
    plt.figure(figsize=(8, 6))
    plt.imshow(frame_rgb)
    plt.title(f"Frame {frame_index} — Circular electrode candidates")
    plt.axis('off')

    for (cx, cy) in electrode_centroids:
        plt.scatter(cx, cy, c='red', s=60)

    plt.show()


    frame_index += 1



cap.release()
print("Circularity filtering completed.")
