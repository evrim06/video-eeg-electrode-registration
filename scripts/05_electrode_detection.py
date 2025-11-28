import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------------------------------------------------
# Hyper-parameters (easy to tweak)
# ---------------------------------------------------------
BRIGHT_THRESH = 180     # how bright a pixel must be to count as "metal"
MIN_AREA      = 30      # min blob size
MAX_AREA      = 900     # max blob size
MIN_CIRC      = 0.45    # how round it must be (0–1)
CAP_TOP_RATIO = 0.0     # use full cap height (0 = top, 1 = bottom)
CAP_BOT_RATIO = 0.7     # ignore bottom 30% (face/strap)


# ---------------------------------------------------------
# 1) Electrode detection inside CAP region (brightness + shape)
# ---------------------------------------------------------
def detect_electrodes_in_cap(frame_rgb, cap_mask):
    """
    1. Restrict to cap mask
    2. Restrict vertically to top 70% of cap (ignore face/strap)
    3. Threshold bright metallic blobs
    4. Filter by size and circularity
    """

    h, w = cap_mask.shape

    # --- bounding box of the cap mask ---
    ys, xs = np.where(cap_mask > 0)
    if len(ys) == 0:
        return []

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # vertical crop within the cap (ignore bottom part where face is)
    cap_height = y_max - y_min
    y_top = int(y_min + CAP_TOP_RATIO * cap_height)
    y_bot = int(y_min + CAP_BOT_RATIO * cap_height)

    # 1) grayscale
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # 2) apply cap mask + vertical restriction
    mask_uint8 = cap_mask.astype(np.uint8)
    masked = cv2.bitwise_and(gray, gray, mask=mask_uint8)

    roi = np.zeros_like(masked)
    roi[y_top:y_bot, x_min:x_max] = masked[y_top:y_bot, x_min:x_max]

    # 3) a bit of smoothing + contrast
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    roi_eq   = cv2.equalizeHist(roi_blur)

    # 4) brightness threshold
    _, thresh = cv2.threshold(roi_eq, BRIGHT_THRESH, 255, cv2.THRESH_BINARY)

    # 5) small morphology to remove noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6) connected components
    labeled, num_labels = label(thresh)
    centroids = []

    for region_id in range(1, num_labels + 1):
        region = (labeled == region_id)
        area   = region.sum()

        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        region_uint8 = (region.astype(np.uint8) * 255)
        contours, _  = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRC:
            continue

        ys_r, xs_r = np.where(region)
        cx, cy = xs_r.mean(), ys_r.mean()
        centroids.append((int(cx), int(cy)))

    return centroids


# ---------------------------------------------------------
# 2) Load SAM2 predictor
# ---------------------------------------------------------
predictor = SAM2ImagePredictor.from_pretrained(
    "facebook/sam2-hiera-large",
    device="cpu"
)


# ---------------------------------------------------------
# 3) Load video & choose ROI on first frame
# ---------------------------------------------------------
video_path = "data/IMG_2763.mp4"
cap = cv2.VideoCapture(video_path)

ret, first_frame = cap.read()
if not ret:
    print("Could not read first frame")
    cap.release()
    raise SystemExit

first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(first_rgb)
plt.title("Click TOP-LEFT and BOTTOM-RIGHT of the cap ROI")
pts = plt.ginput(2)
plt.close()

(x1, y1), (x2, y2) = pts
x, y = int(min(x1, x2)), int(min(y1, y2))
w, h = int(abs(x2 - x1)), int(abs(y2 - y1))
print(f"ROI: x={x}, y={y}, w={w}, h={h}")

# rewind video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# ---------------------------------------------------------
# 4) Process first N frames
# ---------------------------------------------------------
N_FRAMES = 3
frame_idx = 0

while frame_idx < N_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- SAM2 cap segmentation with fixed box ---
    predictor.set_image(frame_rgb)
    masks, scores, logits = predictor.predict(
        box=np.array([[x, y, x + w, y + h]]),
        multimask_output=False
    )
    cap_mask = masks[0].astype(bool)

    # --- electrode detection inside this cap mask ---
    centroids = detect_electrodes_in_cap(frame_rgb, cap_mask)
    print(f"Frame {frame_idx}: detected {len(centroids)} electrodes")

    # --- visualize ---
    plt.figure(figsize=(8, 6))
    plt.imshow(frame_rgb)
    plt.title(f"Frame {frame_idx} — Electrode Candidates")
    plt.axis("off")

    # ROI box
    plt.plot([x, x + w, x + w, x, x],
             [y, y, y + h, y + h, y],
             '--', color='yellow')

    # electrode centroids
    for (cx, cy) in centroids:
        plt.scatter(cx, cy, c="red", s=50)

    plt.show()

    frame_idx += 1

cap.release()
print("Done.")
