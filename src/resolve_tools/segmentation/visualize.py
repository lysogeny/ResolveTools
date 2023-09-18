import numpy as np
from skimage import exposure, segmentation
import colorsys

from ..image.utils import claher
from .postprocess import _intersection_over_union

##############################
### Outline to Grayscale Image
##############################

def add_mask_outline_to_grayscale(gray, mask, rgb=True):
    """ Add red cell outline to grayscale image if rgb,
        else white/black outline on gray image.
    """
    if grb:
        img = np.repeat(claher(gray)[...,None],3,axis=2)
        outline = segmentation.find_boundaries(mask)
        img[outline] = np.asarray([1,0,0])
        return img
    else:
        img = gray.copy()
        outline = segmentation.find_boundaries(mask, mode="inner")
        img[outline] = 0
        outline = segmentation.find_boundaries(mask, mode="outer")
        img[outline] = 1
        return img

def add_mask_outline_confidence_to_grayscale(gray, mask, mask_confident, ioucutoff=0.8):
    """ Add red cell outline to grayscale image,
        green if cell has correspondence in confident mask.
    """
    img = np.repeat(claher(gray)[...,None],3,axis=2)
    outline = segmentation.find_boundaries(mask)
    img[outline] = np.asarray([0,1,0])
    repl = np.arange(mask.max()+1).astype(int)
    iou = _intersection_over_union(mask, mask_confident)
    repl[iou.max(axis=1)>=ioucutoff] = 0
    outline = segmentation.find_boundaries(repl[mask])
    img[outline] = np.asarray([1,0,0])
    return img

##############################
### RGB mask to grayscale
##############################

def get_rgb_distinct(N):
    """ Return N distinct colors.
    """
    hues = np.linspace(0, 1, N+1)[np.random.permutation(N)]
    cols = np.asarray([colorsys.hsv_to_rgb(h,np.random.uniform(0.5,1),1) for h in hues])
    return cols

def add_mask_rgb_to_grayscale_ON2(gray, mask, cols_=None, verbose=False):
    """ Add colored cell masks to grayscale image, slow version.
    """
    img_gray = claher(gray)
    img = np.repeat(img_gray[...,None],3,axis=2)
    cs = np.unique(mask)
    cs = cs[cs!=0]
    ons = np.ones((1,3))
    cols = cols_ if cols_ is not None else get_rgb_distinct(cs.shape[0])
    shw = lambda x: tqdm(x) if verbose else x
    for i, c in shw(enumerate(cs)):
        m = mask==c
        img[m] = 0.2*img_gray[m][:,None]*ons + 0.6*img_gray[m][:,None]*cols[i][None] + 0.2*cols[i]
    outline = segmentation.find_boundaries(mask)
    img[outline] = np.asarray([1,1,1])
    return img

def add_mask_rgb_to_grayscale(gray, mask, cols_=None):
    """ Add colored cell masks to grayscale image.
    """
    img_gray = claher(gray)
    img = np.repeat(img_gray[...,None],3,axis=2)
    ons = np.ones((1,3))
    cols = cols_ if cols_ is not None else get_rgb_distinct(mask.max())
    nmask = mask!=0
    img[nmask] = 0.2*img[nmask] + (0.6*img_gray[nmask][:,None] + 0.2)*cols[mask[nmask]-1]
    outline = segmentation.find_boundaries(mask)
    img[outline] = np.asarray([1,1,1])
    return img

