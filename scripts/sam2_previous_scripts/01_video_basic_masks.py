from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Load SAM2 with custom parameters
mask_generator  = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2-hiera-large",
    device="cpu",
    points_per_side=48,
    crop_n_layers=1,
    crop_overlap_ratio=512/1500,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.85,
    min_mask_region_area=0,
    box_nms_thresh=0.9,
    output_mode="binary_mask"
)

#Open video file
video_path = "data/IMG_2763.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
print("Video opened successfully.")

#Process first 5 frames
frame_count = 0

while frame_count < 5:
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read the frame {frame_count}.")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"/nProcessing frame {frame_count}...")
    masks = mask_generator.generate(frame_rgb)
    print("Number of masks:", len(masks))

    plt.figure(figsize=(8,6))
    plt.imshow(frame_rgb)
    plt.title(f"Frame {frame_count} with {len(masks)} masks")
    plt.axis('off')
    plt.show()

    frame_count += 1
cap.release()  
print("Video processing completed.")