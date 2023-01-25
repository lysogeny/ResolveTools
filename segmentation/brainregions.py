import numpy as np
from skimage.segmentation import expand_labels
import cv2
from PIL import Image
from matplotlib import image as mimage
import re
import os
from matplotlib.colors import rgb2hex

from .visualize import get_rgb_distinct
from ..image.utils import get_single_modality_shape
from .counts import read_loom
from ..utils.utils import printwtime
from ..utils.parameters import CONFOCAL_VOXEL_SIZE


##############################
### Brain Regions Initial
##############################

regionkey = {   0 : "unknown",
                1 : "CTX IS", # Cortex, Injection Site
                2 : "CC IS", # Corpus Callosum, Injection Site
                3 : "SN IS", # Septal Nuclei, Injection Site
                4 : "LV IS", # Lateral Ventricle, Injection Site
                5 : "STR IS", # Striatum, Injection Site
                
                6 : "CTX CLH", # Cortex, Contralateral Hemisphere
                7 : "CC CLH", # Corpus Callosum, Contralateral Hemisphere
                8 : "SN CLH", # Septal Nuclei, Contralateral Hemisphere
                9 : "LV CLH", # Lateral Ventricle, Contralateral Hemisphere
               11 : "STR CLH", # Striatum, Contralateral Hemisphere
               
               12 : "TM IS", # Tumor as Injection Site
               13 : "TM", # Tumor
               14 : "TM/LV", # Tumor and Lateral Ventricle
               
               15 : "CTX", # Cortex
               16 : "CC", # Corpus Callosum
               17 : "SN", # Septal Nuclei
               18 : "LV", # Lateral Ventricle
               19 : "STR" # Striatum
             }

#(get_rgb_distinct(30)*255).astype(int)
regioncolors = np.asarray([[  0,   0,   0],
                           [ 74, 161, 255],
                           [ 65,  78, 255],
                           [ 82, 255, 165],
                           [ 14, 255,  80],
                           [160, 255, 111],
                           [218, 255, 105],
                           [ 35, 186, 255],
                           [ 67, 255,  80],
                           [223, 125, 255],
                           [255, 164,  99],
                           [ 39, 255, 232],
                           [255,  72,  72],
                           [ 57, 111, 255],
                           [  5, 229, 255],
                           [ 12, 255, 179],
                           [255,  94,  52],
                           [247,  51, 255],
                           [255, 120, 176],
                           [255,  24,  72],
                           [123,  54, 255],
                           [173,  73, 255],
                           [255,  60, 181],
                           [ 99, 255,  74],
                           [247, 255,  34],
                           [ 40,   6, 255],
                           [255,  23,  23],
                           [255,  82, 225],
                           [159, 255,  42],
                           [255, 181,  61],
                           [255, 221,  60]])

#"_nl_".join([rgb2hex(x).replace("#","ff") for x in regioncolors/255])
#ff000000_nl_ff4aa1ff_nl_ff414eff_nl_ff52ffa5_nl_ff0eff50_nl_ffa0ff6f_nl_ffdaff69_nl_ff23baff_nl_ff43ff50_nl_ffdf7dff_nl_ffffa463_nl_ff27ffe8_nl_ffff4848_nl_ff396fff_nl_ff05e5ff_nl_ff0cffb3_nl_ffff5e34_nl_fff733ff_nl_ffff78b0_nl_ffff1848_nl_ff7b36ff_nl_ffad49ff_nl_ffff3cb5_nl_ff63ff4a_nl_fff7ff22_nl_ff2806ff_nl_ffff1717_nl_ffff52e1_nl_ff9fff2a_nl_ffffb53d_nl_ffffdd3c


def processes_regionsegmentation_initial(imagepath, referenceimg, annotpath, regioncolors = regioncolors):
    """ Processes initial color region segmentation with key, save region mask.
    """
    #image = Image.open(imagepath)
    #img = np.asarray(image)
    img = (mimage.imread(imagepath)[...,:3]*255).astype(int)
    assert len(img.shape)==3
    
    targetshape = get_single_modality_shape(referenceimg)[1:]
    #mask = np.load(maskpath)[maskkey]

    uniques = [list(np.unique(img[...,i])) for i in range(img.shape[-1])]
    def maybe_present(color, uniques):
        return np.all([color[i] in uniques[i] for i in range(len(color))])
    color_mask = [maybe_present(color, uniques) for color in regioncolors[1:]]
    maybe_colors = regioncolors[1:][color_mask]
    maybe_colors_ind = np.arange(len(regioncolors[1:]))[color_mask]+1

    regions = np.zeros(img.shape[:2], dtype=int)
    for i, color in zip(maybe_colors_ind, maybe_colors):
        regions[np.all(np.logical_and(img<=color+2, img>=color-2), axis=-1)] = i
    if (regions==0).sum()/np.prod(regions.shape)>0.01:
        raise ValueError("Something didn't work right, found more than 1% of annotation empty!")

    regions = expand_labels(regions, 10)
    regions = cv2.resize(regions, targetshape[::-1], interpolation = cv2.INTER_NEAREST)
    
    np.savez_compressed(annotpath, regions=regions)

##############################
### Brain Regions Of Resolve Counts
##############################

def add_regions_to_resolve(pathin, pathout, pathregion, verbose=True):
    """ Add region column to registered resolve transcript list.
    """
    rim = ResolveImage(pathin)
    regions = np.load(pathregion)["regions"]
    rim.full_data["Region"] = regions[rim.full_data["y"], rim.full_data["x"]].astype(int)
    rim.full_data.to_csv(pathout, index=False)
    if verbose: print(np.unique(rim.full_data["Region"], return_counts=True))

##############################
### Brain Regions to AnnData
##############################

def add_region_toadata(adatafile, regionfile, roicheck=True):
    """ Load adata from adatafile, add regions from regionfile, save to same file.
    """
    adata = read_loom(adatafile)
    if not ("x" in adata.obs.columns and "y" in adata.obs.columns):
        printwtime("Found no position information for {}!".format(adatafile))
        return None
    roi = re.search(r'R(\d)_W(\d)A(\d)', os.path.basename(regionfile)).group(0)
    rois = np.unique(adata.obs["ROI"])
    if roicheck and not (len(rois)==1 and roi==rois[0]):
        printwtime("Wrong ROI combination for {}!".format(adatafile))
        return None
    
    regions = np.load(regionfile)["regions"]
    yIm = (adata.obs["y"]/CONFOCAL_VOXEL_SIZE[1]).astype(int)
    xIm = (adata.obs["x"]/CONFOCAL_VOXEL_SIZE[2]).astype(int)
    if yIm.min()<0 or xIm.min()<0 or yIm.max()>regions.shape[0] or xIm.max()>regions.shape[1]:
        printwtime("Entries out of bounds for {}!".format(adatafile))
        return None
    
    adata.obs["BrainRegion"] = regions[yIm, xIm]
    adata.obs["BrainRegionName"] = adata.obs["BrainRegion"].apply(lambda x: regionkey[x])
    adata.write_loom(adatafile)