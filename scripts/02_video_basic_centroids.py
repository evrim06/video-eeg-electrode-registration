import numpy as np
from scipy.ndimage import label

def get_small_masks_and_centroids(mask, min_size=50, max_size=3000):
    """
    Split a SAM2 mask into connected components
    and return small components with their centroids.
    """

    # SAM2 mask is boolean → convert to uint8
    mask_uint = mask.astype(np.uint8)

    # Connected component labeling
    labeled, num_features = label(mask_uint)

    results = []

    for region_id in range(1, num_features + 1):
        region_mask = (labeled == region_id)
        size = region_mask.sum()

        # Filter by blob size (electrode-sized)
        if min_size <= size <= max_size:
            coords = np.argwhere(region_mask)
            cy, cx = coords.mean(axis=0)  # row, col
            centroid = (int(cx), int(cy))  # return (x, y)
            results.append((region_mask, centroid))

    return results
