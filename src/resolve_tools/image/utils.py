##############################
### Reading images
##############################

import cv2
from tifffile import TiffFile

def read_single_modality_confocal(file, norm=True):
    """ Read single modality confocal .tif image.
    """
    img = np.asarray(cv2.imreadmulti(file, flags=cv2.IMREAD_ANYDEPTH )[1])
    if norm:
        return img/img.max()
    else:
        return img

def get_single_modality_shape(file):
    """ Get shape of single modality confocal .tif image,
        without actually reading the image.
        [z,y,x]
    """
    tif = TiffFile(file)
    shape = [len(tif.pages)] + list(tif.pages[0].shape)
    return shape

##############################
### Saving images
##############################

from tifffile import imwrite

def save_tiff(filename, data, compression="zlib"):
    """ Save image stack as tif.
    """
    imwrite(filename, data, metadata={"mode": "composite"}, imagej=True, compression=compression)

def save_tiff_from_float(filename, data, compression="zlib"):
    """ save_tiff, with additional converion of the image from [0,1] to 8bit.
    """
    imwrite(filename, (data*255).astype("H"), metadata={"mode": "composite"}, imagej=True, compression=compression)

##############################
### Modifying images
##############################

from skimage import exposure
import mclahe as mc

def claher(img, kernel_size = 127, clip_limit = 0.01, nbins = 256):
    """
    Runs Contrast Limited Adaptive Histogram Equalization (CLAHE) and normalizes to [0,1].
    """
    img = exposure.equalize_adapthist(img, kernel_size = kernel_size, clip_limit = clip_limit, nbins = nbins)
    img = img / img.max()
    return img

def mclaher(img, kernel_size=[10,128,128],n_bins=128,clip_limit=0.01,adaptive_hist_range=False):
    """
    Runs Contrast Limited Adaptive Histogram Equalization (CLAHE) in 3D and normalizes to [0,1].
    """
    img = mc.mclahe(img, kernel_size=kernel_size, n_bins=n_bins, clip_limit=clip_limit, adaptive_hist_range=adaptive_hist_range)
    img = img / img.max()
    return img

import cv2
import numpy as np

def resize_shrink(img, factor):
    """ Shrink image by factor along all axes.
    """
    return cv2.resize(img, np.asarray(img.shape)[::-1]//factor, interpolation = cv2.INTER_AREA)



