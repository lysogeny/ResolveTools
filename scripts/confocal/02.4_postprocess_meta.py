import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm

import os
import sys
sys.path.insert(0,'/data/')

from resolve_tools.segmentation.utils import segmentation_to_meta_df, save_sampled_boundary
from skimage.segmentation import find_boundaries
from resolve_tools.utils.utils import printwtime
import re

filename = sys.argv[1]

printwtime("Loading Mask")
file = np.load("/data/confocal/"+filename)
mask = file["mask_post"]

roikey = re.search(r'R(\d)_W(\d)A(\d)', filename).group(0)
df = segmentation_to_meta_df(mask, roikey = roikey)
df.to_csv(filename.replace("post.npz", "post_meta.csv"))

inner = find_boundaries(mask, mode="inner")
save_sampled_boundary(mask, inner, filename.replace("post.npz", "post_boundary.npz"), N=60, seed=42)
