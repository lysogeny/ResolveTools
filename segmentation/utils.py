import numpy as np
from scipy.ndimage import distance_transform_edt

##############################
### Modify Segmentation
##############################

def expand_labels(label_image, distance=1, sampling=None):
    """ Scipy function, but with anisotropic sampling enabled.
    """
    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True, sampling = sampling
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out