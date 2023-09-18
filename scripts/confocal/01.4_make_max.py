import os
from tqdm import tqdm

import sys
sys.path.insert(0,'/data/')
from resolve_tools.image.utils import save_tiff, save_tiff_from_float, claher, read_single_modality_confocal
from resolve_tools.utils.utils import printwtime

path = "/data/confocal/01_image/"
images = os.listdir(path)
#images_nmax = list(filter(lambda x: ("DAPI" in x or "WGA" in x) and "_max" not in x,images))
images_nmax = list(filter(lambda x: "_max" not in x,images))
images_makemax = list(filter(lambda x: x.replace(".tif","_max_claher.tif") not in images,images_nmax))

for image in tqdm(images_makemax):
    printwtime(image)
    img = read_single_modality_confocal(path+image, norm=False)
    save_tiff(path+image.replace(".tif","_max.tif"), img.max(axis=0)[None])
    save_tiff_from_float(path+image.replace(".tif","_max_claher.tif"), claher(img.max(axis=0))[None])
    del img
