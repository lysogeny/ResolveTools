import os
from tqdm import tqdm
import re

import sys
sys.path.insert(0,'/data/')
from resolve_tools.segmentation.brainregions import processes_regionsegmentation_initial
from resolve_tools.utils.utils import printwtime

path = "/data/confocal/03_annotation/"
process = list(filter(lambda x: "_annotation.png" in x, os.listdir("/data/confocal/03_annotation")))

for image in process:
    printwtime("Processing annotation {}.".format(image))
    roi = re.search(r'R(\d)_W(\d)A(\d)', image).group(0)
    imagepath = path+image
    referenceimg = "/data/confocal/01_image/Confocal_"+roi+"_DAPI.tif"
    annotpath = path+"Confocal_"+roi+"_annotated_regions.npz"
    
    processes_regionsegmentation_initial(imagepath, referenceimg, annotpath)
