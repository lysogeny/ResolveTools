import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm

import os
import sys
sys.path.insert(0,'/data/')

from resolve_tools.image.utils import save_tiff, claher, read_single_modality_confocal
from resolve_tools.image.lif import LifFile, read_lif_image
from resolve_tools.registration.registerSIFT import find_homography, warp_image, transform_coordinate, scale_homography,\
                                                    get_transformed_corners
from resolve_tools.segmentation.brainregions import processes_regionsegmentation_initial
from resolve_tools.utils.utils import printwtime

def find_corners():
    printwtime("Loading Images")
    
    scaleD, scaleA = 6, 4
    
    dapimaxname = sys.argv[1]
    
    dapi_max_full = cv2.imread('/data/confocal_raw/'+dapimaxname+'_DAPI_max.tif', cv2.IMREAD_ANYDEPTH)
    dapi_max = claher(cv2.resize(dapi_max_full, np.asarray(dapi_max_full.shape)[::-1]//scaleD, interpolation = cv2.INTER_AREA))
    
    Aname = sys.argv[2]
    
    A_full = cv2.imread('/data/resolve/01_raw/'+Aname+'_DAPI.tiff', cv2.IMREAD_ANYDEPTH)
    if int(sys.argv[3])>0:
        A_full = np.flip(A_full,0)
    A = claher(cv2.resize(A_full, np.asarray(A_full.shape)[::-1]//scaleA, interpolation = cv2.INTER_AREA))
    
    homography_fromA, homography_toA = find_homography(dapi_max, A, verbose=True)
    homography_fromA_full = scale_homography(homography_fromA, scaleD, scaleA)
    corners_Afull = np.reshape(np.asarray([[transform_coordinate(homography_fromA_full,i,j) for j in [0,A_full.shape[0]]] for i in [0,A_full.shape[1]]]), (4,2))
    corners_Afull = corners_Afull[[0,1,3,2,0]]
    np.savez("/data/confocal_raw/"+dapimaxname+"__"+Aname+"__corners.npz", corners_Afull=corners_Afull)
    
    printwtime("Saving Plot")
    fig, ax = plt.subplots(1,3,figsize=(10,10))
    ax[0].imshow(dapi_max,cmap="gray")
    ax[1].imshow(warp_image(dapi_max, A, homography_fromA),cmap="gray")
    ax[2].imshow(A,cmap="gray")
    plt.savefig('/data/confocal_raw/'+dapimaxname+"__"+Aname+"__prelaligned.pdf")
    plt.close()

partsorderreporter = ["DAPI", "mCherry", "EGFP", "WGA"]
partsordernaive  =   ["DAPI", "IB4",     "HCM",  "WGA"]

def write_parts(partsorder=partsorderreporter):
    dapimaxname = sys.argv[1]
    Aname = sys.argv[2]
    
    cfile = np.load("/data/confocal_raw/"+dapimaxname+"__"+Aname+"__corners.npz")
    corners_Afull = cfile["corners_Afull"]

    file = LifFile('/data/confocal_raw/'+dapimaxname+".lif")
    lif = file.get_image(0)

    def bounds_from_corners(corners, add=100):
        return np.asarray([[max(0, corners[:,0].min()-add), corners[:,0].max()+add],
                          [max(0, corners[:,1].min()-add), corners[:,1].max()+add]]).astype(int)
    Abounds = bounds_from_corners(corners_Afull)

    def apply_bounds(arr, bounds):
        return arr[bounds[1,0]:bounds[1,1], bounds[0,0]:bounds[0,1]]
    
    for j in range(4):
        printwtime("Saving Clipped Modality "+partsorder[j])
        if len(sys.argv)<=5:
            img = np.asarray([apply_bounds(np.asarray(read_lif_image(lif, i, j)), Abounds)[None] for i in tqdm(range(lif.info["dims"][2]))])
        else:
            img = np.asarray([apply_bounds(np.asarray(read_lif_image(lif, i, j)), Abounds)[None] for i in tqdm(range(int(sys.argv[5]),int(sys.argv[6])))])
        save_tiff('/data/confocal/01_image/Confocal_'+Aname.replace("T6GBM_", "")+"_"+partsorder[j]+'.tif', img)
        del img

printwtime("Registering "+sys.argv[2]+" to "+sys.argv[1]+".")
find_corners()
write_parts(partsorderreporter if sys.argv[4]=="reporter" else partsordernaive)
