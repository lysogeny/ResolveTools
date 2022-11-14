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

def expand_labels_tiled(label_image, tilesize=2000, distance=1, sampling=None):
    """ expand_labels, but for tiles and stitched together to save memory.
        !! Will break if overlap includes more than one neighboring tile,
           but I won't fix this since this use case would be stupid. !!
    """
    overlap = int(1.5*distance) if sampling is None else int(1.5*distance/min(sampling))
    if overlap>tilesize: raise ValueError("Tilesize is too small for overlap!")
    Nx = int(np.ceil(label_image.shape[-1]/tilesize))
    Ny = int(np.ceil(label_image.shape[-2]/tilesize))
    expanded = label_image.copy()
    for i in range(Nx):
        for j in range(Ny):
            inputslice = (..., slice(max(0, j*tilesize - overlap), (j+1)*tilesize + overlap, None),
                               slice(max(0, i*tilesize - overlap), (i+1)*tilesize + overlap, None))
            outputslice = (..., slice(0 if j==0 else overlap, tilesize + (0 if j==0 else overlap), None),
                                slice(0 if i==0 else overlap, tilesize + (0 if i==0 else overlap), None))
            assignslice = (..., slice(j*tilesize, (j+1)*tilesize, None),
                                slice(i*tilesize, (i+1)*tilesize, None))
            
            expanded[assignslice] = expand_labels(label_image[inputslice], distance=distance, sampling=sampling)[outputslice]
    return expanded
