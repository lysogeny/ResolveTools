##############################
### Reading images
##############################

import cv2

def read_single_modality_confocal(file, norm=True):
    """ Read single modality confocal .tif image.
    """
    img = np.asarray(cv2.imreadmulti(file, flags=cv2.IMREAD_ANYDEPTH )[1])
    if norm:
        return img/img.max()
    else:
        return img

##############################
### Saving images
##############################

from tifffile import imwrite

def save_tiff(filename, data, compression="zlib"):
    """ Save image stack as tif.
    """
    imwrite(filename, data, metadata={"mode": "composite"}, imagej=True, compression=compression)

##############################
### Modifying images
##############################

from skimage import exposure

def claher(img):
    """
    Runs Contrast Limited Adaptive Histogram Equalization (CLAHE) and normalizes to [0,1].
    """
    img = exposure.equalize_adapthist(img, kernel_size = 127, clip_limit = 0.01, nbins = 256)
    img = img / img.max()
    return img

import cv2
import numpy as np

def resize_shrink(img, factor):
    """ Shrink image ba factor along all axes.
    """
    return cv2.resize(img, np.asarray(img.shape)[::-1]//factor, interpolation = cv2.INTER_AREA)