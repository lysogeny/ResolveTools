import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm

import os
import sys
sys.path.insert(0,'/data/')

from resolve_tools.segmentation.postprocess import postprocess_raw_mesmer_masks
from resolve_tools.utils.utils import printwtime

printwtime("Loading Mask")
file = np.load("/data/confocal/"+sys.argv[1])
mask = file["mask_full"]

mask_post = postprocess_raw_mesmer_masks(mask)

printwtime("Saving Mask")
np.savez_compressed("/data/confocal/"+sys.argv[1].replace(".npz","_post.npz"), mask_post=mask_post)
