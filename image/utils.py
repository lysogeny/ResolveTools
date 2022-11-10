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


from readlif.reader import LifFile

def read_lif_image(lif, z, channel):
    """ lif.get_frame apparently uses random ordering, but this works.
    """
    return lif._get_item(z*lif.channels + channel)