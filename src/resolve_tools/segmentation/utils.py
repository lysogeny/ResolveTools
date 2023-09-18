import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
import pandas as pd
import itertools

from .brainregions import regionkey
from ..utils.parameters import CONFOCAL_VOXEL_SIZE

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
        Will probably break if overlap includes more than one neighboring tile,
        but I won't fix this since this use case would be stupid.
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

##############################
### Cell Utils
##############################

def region_to_label(mask, region):
    """ Given mask and RegionProperty from regionprops,
        return label of region.
    """
    return mask[region.slice][region.image][0]

def region_to_brainregion(regionmask, region):
    """ Given brain regionmask and RegionProperty from regionprops,
        return brainregion with largest overlap with region.
    """
    u, c = np.unique(regionmask[region.slice[1:]][np.any(region.image, axis=0)], return_counts=True)
    return u[c.argsort()[::-1]][0]

def region_to_centroid(region, sampling=CONFOCAL_VOXEL_SIZE[:3]):
    """ Given RegionProperty from regionprops, returns centroid in um.
    """
    return region.centroid*np.asarray(sampling)

def region_to_volume(region, sampling=CONFOCAL_VOXEL_SIZE[:3]):
    """ Given RegionProperty from regionprops, returns volume in um.
    """
    return region.image.sum()*np.prod(sampling)

def segmentation_to_meta_df(mask, regionmask = None, roikey = "ROI", sampling=CONFOCAL_VOXEL_SIZE[:3]):
    """ Takes cell segmention and brain region segmentation,
        return meta dataframe for the cells.
    """
    regions = regionprops(mask)
    
    df = pd.DataFrame()
    df["Label"] = [region_to_label(mask, region) for region in regions]
    df["Label"] = df["Label"].astype(int)
    if regionmask is not None:
        df["BrainRegion"] = [region_to_brainregion(regionmask, region) for region in regions]
        df["BrainRegion"] = df["BrainRegion"].astype(int)
        df["BrainRegionName"] = [regionkey[reg][0] for reg in df["BrainRegion"]]
    else:
        df["BrainRegion"] = 0
        df["BrainRegionName"] = ""
    df["ROI"] = roikey
    df.index = df["ROI"]+"_"+df["Label"].astype(str)
    df[["z","y","x"]] =          [region_to_centroid(region, sampling=sampling) for region in regions]
    df[["zImg","yImg","xImg"]] = [region_to_centroid(region, sampling=[1,1,1]) for region in regions]
    df["Volume"] =               [region_to_volume(region, sampling=sampling) for region in regions]
    df["VolumeImg"] =            [region_to_volume(region, sampling=[1,1,1]) for region in regions]
    
    ### ADD CONNECTIVITY?
    
    return df

##############################
### Sample Boundary
##############################

def save_sampled_boundary(mask, inner, file, N=60, sampling=CONFOCAL_VOXEL_SIZE, seed=42):
    """ Get all boundary points for cells in mask, sample N per z plane and save them.
        N=60 is sufficient for only negligible differences to full boundary.
        
        Can set random seed for reproducibility.
    """
    np.random.seed(seed)
    df = pd.DataFrame(np.argwhere(inner), columns=["z","y","x"])
    df["MaskIndex"] = mask[df["z"], df["y"], df["x"]]
    df[["z","y","x"]] = df[["z","y","x"]]*np.asarray(sampling[:3])
    def sample(x, N=60):
        ind = x.index
        return np.random.choice(list(ind), size=N, replace=False) if len(ind)>N else list(ind)
    def sampleallz(x, N=60):
        vals = x.groupby("z").apply(lambda x: sample(x, N))
        return list(vals)
    cells = df.groupby("MaskIndex").apply(lambda x: sampleallz(x))
    cells = cells.apply(lambda x: list(itertools.chain.from_iterable(x)))
    index = list(cells.index)
    order = "xyz"
    fullpoints = np.asarray(df[["x","y","z"]])
    points = np.asarray([fullpoints[cell] for cell in cells], dtype=object)
    print("Total boundary points:  ",len(df))
    print("Sampled boundary points:",sum([len(p) for p in points]))
    np.savez_compressed(file, index=index, order=order, points=points)
