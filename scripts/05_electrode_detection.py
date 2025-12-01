from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import os

# ---------------------------------------------------------
# Helper: extract small round blobs (electrode-like) from one SAM2 mask
# ---------------------------------------------------------
def get_electrode_blobs(mask, min_size=60, max_size=2500, min_circularity=0.35):
    """
    mask: boolean or 0/1 array (H, W)
    returns list of (cx, cy) centroids in mask coordinates
    """
    mask_uint = (mask > 0).astype(np.uint8)
    labeled, num_features = label(mask_uint)

    centroids = []

    for region_id in range(1, num_features + 1):
        region = (labeled == region_id)
        area = region.sum()
        if not (min_size <= area <= max_size):
            continue

        region_uint8 = (region.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        ys, xs = np.where(region)
        cx, cy = xs.mean(), ys.mean()
        centroids.append((int(cx), int(cy)))

    return centroids


# ---------------------------------------------------------
# Load SAM2 Automatic Mask Generator (used for electrodes)
# ---------------------------------------------------------
mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2-hiera-large",
    device="cpu",
    points_per_side=64,         # dense grid → better for small objects
    crop_n_layers=1,
    crop_overlap_ratio=512/1500,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.9,
    min_mask_region_area=5,
    box_nms_thresh=0.9,
    output_mode="binary_mask",
)


# ---------------------------------------------------------
# Load video
# ---------------------------------------------------------
video_path = "data/IMG_2763.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: cannot open video:", video_path)
    raise SystemExit

ret, first_frame = cap.read()
if not ret:
    print("Error: cannot read first frame")
    cap.release()
    raise SystemExit

first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------
# Step 1: ROI selection on first frame (Matplotlib GUI)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(first_rgb)
plt.title("Click TOP-LEFT and BOTTOM-RIGHT of EEG cap ROI")
plt.axis("on")
pts = plt.ginput(2)    # user clicks 2 points
plt.close()

(x1, y1), (x2, y2) = pts
x = int(min(x1, x2))
y = int(min(y1, y2))
w = int(abs(x2 - x1))
h = int(abs(y2 - y1))

print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Create output folder
os.makedirs("results", exist_ok=True)

# To store detections
all_frame_centroids = []   # list of lists of (x, y)

# ---------------------------------------------------------
# Step 2: Process first N_FRAMES using SAM2 inside ROI
# ---------------------------------------------------------
N_FRAMES = 5
frame_idx = 0

while frame_idx < N_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_idx}")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Crop ROI around the cap
    roi_rgb = frame_rgb[y:y+h, x:x+w]

    print(f"\n➡ Processing frame {frame_idx} inside ROI...")
    masks = mask_generator.generate(roi_rgb)
    print("SAM2 masks inside ROI:", len(masks))

    frame_centroids = []

    for m in masks:
        seg = m["segmentation"]      # boolean (H_roi, W_roi)
        blobs = get_electrode_blobs(seg)

        for (cx_roi, cy_roi) in blobs:
            # Convert ROI coords → full-image coords
            cx_full = x + cx_roi
            cy_full = y + cy_roi
            frame_centroids.append((cx_full, cy_full))

    all_frame_centroids.append(frame_centroids)
    print(f"Detected {len(frame_centroids)} electrode-like blobs")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(frame_rgb)
    plt.title(f"Frame {frame_idx} — Electrode candidates")
    plt.axis("off")

    # Draw ROI
    plt.plot([x, x+w, x+w, x, x],
             [y, y, y+h, y+h, y],
             "--", color="yellow")

    # Draw centroids
    for (cx, cy) in frame_centroids:
        plt.scatter(cx, cy, c="red", s=50)

    plt.show()

    frame_idx += 1

cap.release()
print("Done electrode detection.")

# Optional: save detections + ROI to .npz for later steps
np.savez(
    "results/electrodes_sam2_roi.npz",
    roi=np.array([x, y, w, h]),
    centroids=object().__class__(all_frame_centroids)  # hack: store as object array
)
print("Saved detections to results/electrodes_sam2_roi.npz")
