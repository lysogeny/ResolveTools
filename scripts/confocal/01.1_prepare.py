import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tifffile

import os
import sys
sys.path.insert(0,'/data/')
from resolve_tools.image.utils import save_tiff
from resolve_tools.image.lif import LifFile, read_lif_image

from tqdm import tqdm

def make_small_max(name, factor=8):
    file = LifFile("/data/confocal_raw/"+name+".lif")
    lif = file.get_image(0)
    
    zimages = []
    dapi_max = np.array(read_lif_image(lif, 0, 0))
    wga_max = np.array(read_lif_image(lif, 0, 3))
    for i in tqdm(range(lif.nz)):
        cimages = []
        for j in range(lif.channels):
            cimages.append(np.array(read_lif_image(lif, i, j).resize((lif.info["dims"][0]//factor,lif.info["dims"][1]//factor))))
        zimages.append(cimages)
        dapi_max = np.asarray([dapi_max, np.array(read_lif_image(lif, i, 0))]).max(axis=0)
        wga_max = np.asarray([wga_max, np.array(read_lif_image(lif, i, 3))]).max(axis=0)
    small = np.asarray(zimages).astype("H")
    save_tiff('/data/confocal_raw/'+name+'_small.tif',small)
    save_tiff('/data/confocal_raw/'+name+'_DAPI_max.tif',dapi_max[None])
    save_tiff('/data/confocal_raw/'+name+'_WGA_max.tif',wga_max[None])


#name = sys.argv[1]

run1 = ["Run1/"+name.replace(".lif","") for name in list(filter(lambda x: ".lif" in x, os.listdir("/data/confocal_raw/Run1/")))
                                                    if name.replace(".lif","_small.tif") not in os.listdir("/data/confocal_raw/Run1/")]
run2 = ["Run2/"+name.replace(".lif","") for name in list(filter(lambda x: ".lif" in x, os.listdir("/data/confocal_raw/Run2/")))
                                                    if name.replace(".lif","_small.tif") not in os.listdir("/data/confocal_raw/Run2/")]

for name in tqdm(run1+run2):
    print(name)
    make_small_max(name)
