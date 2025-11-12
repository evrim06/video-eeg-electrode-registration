from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Force SAM2 to load on CPU
mask_generator  = SAM2AutomaticMaskGenerator.from_pretrained(
    "facebook/sam2-hiera-large",
    device="cpu",
    points_per_side=80,
    crop_n_layers=1,
    crop_overlap_ratio=512/1500,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.85,
    min_mask_region_area=5,
    box_nms_thresh=0.9,
    output_mode="binary_mask"
)



# Read an image
image = cv2.imread("data/example_cap.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image_rgb)


print(len(masks))
print(masks[0].keys())

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 