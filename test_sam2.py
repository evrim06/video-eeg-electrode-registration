from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import matplotlib.pyplot as plt

# Force SAM2 to load on CPU
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device="cpu")

# Read an image
image = cv2.imread("example.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# Choose a prompt point (x, y)
input_point = [[400, 400]]   # pick a point roughly near your object
input_label = [1]

#Run segmentation
masks, scores, logits = predictor.predict( point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# Show result
plt.imshow(image_rgb)
plt.imshow(masks[0], alpha=0.5)
plt.title("SAM2 Segmentation Test")
plt.axis("off")
plt.show()
