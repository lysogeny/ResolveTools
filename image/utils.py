from tifffile import imwrite

def save_tiff(filename, data, compression="zlib"):
    """ Save image stack as tiff.
    """
    imwrite(filename, data, metadata={"mode": "composite"}, imagej=True, compression=compression)


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
    return cv2.resize(target_max, np.asarray(target_max.shape)[::-1]//factor, interpolation = cv2.INTER_AREA)