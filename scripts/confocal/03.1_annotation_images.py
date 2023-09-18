import os
from tqdm import tqdm
import cv2
import numpy as np
import re

import sys
sys.path.insert(0,'/data/')
from resolve_tools.image.utils import resize_shrink, read_single_modality_confocal
from resolve_tools.utils.utils import printwtime

factor = 2
path = "/data/confocal/01_image/"
pathout = "/data/confocal/03_annotation/"
roikeys = np.unique([re.search(r'R(\d)_W(\d)A(\d)', filename).group(0) for filename in os.listdir(path)])

for roi in tqdm(roikeys):
    printwtime(roi)
    
    dapimax = read_single_modality_confocal(path+"Confocal_"+roi+"_DAPI_max_claher.tif")[0]
    wgamax = read_single_modality_confocal(path+"Confocal_"+roi+"_WGA_max_claher.tif")[0]
    
    dapimax = resize_shrink(dapimax, factor)
    dapimax = dapimax/dapimax.max()
    wgamax = resize_shrink(wgamax, factor)
    wgamax = wgamax/wgamax.max()
    
    final = np.repeat(np.zeros(dapimax.shape, dtype=float)[..., None], 3, axis=-1)
    final[..., 2] = dapimax
    final[..., 1] = wgamax
    
    cv2.imwrite(pathout+'Confocal_'+roi+'_annotation_tempalte.jpg', (final*255).astype("H"), [cv2.IMWRITE_JPEG_QUALITY, 100])
