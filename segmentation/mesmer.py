import numpy as np
#import cv2
#import matplotlib.pyplot as plt
#from cellpose import utils
from cellpose.io import imread
#from skimage import exposure, morphology, segmentation
from tqdm import tqdm

from ResolveTools.segmentation.postprocess import postprocess_raw_mesmer_masks
from ResolveTools.segmentation.utils import expand_labels_tiled


from deepcell.applications import Mesmer
app = Mesmer()


dapi_z = imread("Confocal_R1_W7A1_DAPI.tif")
X_data = dapi_z[...,None][...,[0,0]].copy()

segmentation_predictions = np.asarray([app.predict(X_data[i:i+1], image_mpp=0.3, compartment='nuclear', postprocess_kwargs_nuclear={   'maxima_threshold': 0.07, # lower->more cells
                                                                                                                    'maxima_smooth': 0.5,
                                                                                                                    'interior_threshold': 0.4, # lower->larger cells
                                                                                                                    'interior_smooth': 2,
                                                                                                                    'small_objects_threshold': 150,
                                                                                                                    'fill_holes_threshold': 15,
                                                                                                                    'radius': 2}) for i in tqdm(range(X_data.shape[0]))])[:,0,...,0]

segmentation_predictions_conf = np.asarray([app.predict(X_data[i:i+1], image_mpp=0.3, compartment='nuclear', postprocess_kwargs_nuclear={   'maxima_threshold': 0.3, # lower->more cells
                                                                                                                    'maxima_smooth': 0.5,
                                                                                                                    'interior_threshold': 0.4, # lower->larger cells
                                                                                                                    'interior_smooth': 2,
                                                                                                                    'small_objects_threshold': 150,
                                                                                                                    'fill_holes_threshold': 15,
                                                                                                                    'radius': 2}) for i in tqdm(range(X_data.shape[0]))])[:,0,...,0]

np.savez_compressed("Confocal_R1_W7A1_DAPI_mesmer_nuclei.npz",
                    mask_full=segmentation_predictions,
                    mask_confident=segmentation_predictions_conf,
                    description=\
"""Mesmer segmentation with image_mpp=0.3, compartment='nuclear' and postprocessing:
{'maxima_threshold': 0.07,
'maxima_smooth': 0.5,
'interior_threshold': 0.4,
'interior_smooth': 2,
'small_objects_threshold': 150,
'fill_holes_threshold': 15,
'radius': 2}
Confident version uses 'maxima_threshold': 0.3.""")

mask_post = postprocess_raw_mesmer_masks(segmentation_predictions)

np.savez_compressed("Confocal_R1_W7A1_mesmer_nuclei_post.npz", mask_post=mask_post)

mask_expanded = expand_labels_tiled(mask_post, tilesize=2000, distance=10, sampling=[1.,0.142,0.142])

np.savez_compressed("Confocal_R1_W7A1_mesmer_nuclei_post_expanded10um.npz",
                    mask_post_expanded=mask_expanded)


