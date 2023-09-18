import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from PIL import ImageDraw

import os
import sys
sys.path.insert(0,'/data/')

from resolve_tools.image.utils import save_tiff, claher, resize_shrink
from resolve_tools.registration.registerSIFT import find_homography, find_homographies, warp_image,\
                                                    transform_coordinate, scale_homography, get_transformed_corners,\
                                                    homography_shift_target
from resolve_tools.registration.registerTiles import tile_2Dimage
from resolve_tools.utils.utils import printwtime

name = sys.argv[1]

sourcepath = '/data/resolve/01_raw/T6GBM_'+name+'_DAPI.tiff'
flipsource = int(sys.argv[2])==1
targetpath = '/data/confocal/01_image/Confocal_'+name+'_DAPI.tif'
pdfregisterpath = "/data/resolve/03_registration2D/T6GBM_"+name+"_registered2D_tiles.pdf"
pdftranscriptpath = "/data/resolve/03_registration2D/T6GBM_"+name+"_registered2D_transcripts.pdf"
controltifpath = "/data/resolve/03_registration2D/T6GBM_"+name+"_registered2D.tif"
transcriptpath = '/data/resolve/01_raw/T6GBM_'+name+'_transcripts.txt'
transcriptoutpath = "/data/resolve/03_registration2D/T6GBM_"+name+"_transcripts_registered2D.txt"
shrink_factor = 2 if len(sys.argv)==3 else int(sys.argv[3])
TILE_SIZE = 2144 if len(sys.argv)<5 else int(sys.argv[4])
mode = "partialaffine"
emptycutoffmean = 0.01

tile_s_shrunken = TILE_SIZE//shrink_factor

printwtime("Loading Images")

source_full = cv2.imread(sourcepath, cv2.IMREAD_ANYDEPTH)
if flipsource: source_full = np.flip(source_full,0)

target_full = np.asarray(cv2.imreadmulti(targetpath, flags=cv2.IMREAD_ANYDEPTH )[1]).max(axis=0)

target = claher(resize_shrink(target_full, shrink_factor))
source = claher(resize_shrink(source_full, shrink_factor))

source_tiles, slices, Ntiles, Ntilesaxes = tile_2Dimage(source, tile_s_shrunken)

is_empty = np.asarray([t.mean() for t in source_tiles])<emptycutoffmean
not_empty_ind = np.arange(len(is_empty))[~is_empty]

homographies = find_homographies(target, source_tiles, verbose=True, mode=mode)
corners = [get_transformed_corners(s, h[0]) if h[0] is not None else None for h, s in zip(homographies, source_tiles)]
homographies_full = [scale_homography(h[0], shrink_factor, shrink_factor) if h[0] is not None else None for h in homographies]
source_tiles_full, slices_full, Ntiles, Ntilesaxes = tile_2Dimage(source_full, TILE_SIZE)
corners_full = [get_transformed_corners(s, h) if h is not None else None for h, s in zip(homographies_full, source_tiles_full)]

def area(c):
    if c is None: return 0
    first = c[0][0]*c[1][1] + c[1][0]*c[2][1] + c[2][0]*c[3][1] + c[3][0]*c[0][1]
    second = c[1][0]*c[0][1] + c[2][0]*c[1][1] + c[3][0]*c[2][1] + c[0][0]*c[3][1]
    area = -0.5*(first - second)
    return area
areas = np.asarray([area(c) for c in corners_full])

is_empty = np.logical_or(is_empty, np.logical_or(areas/np.median(areas)<0.9, areas/np.median(areas)>1.1))
not_empty_ind = np.arange(len(is_empty))[~is_empty]

fig, ax = plt.subplots(1,1,figsize=(20,20))
ax.imshow(target_full, cmap='gray',origin="lower")
for cor, empty in zip(corners_full, is_empty):
    if not empty:
        ax.plot(cor[:,0],cor[:,1])
plt.savefig(pdfregisterpath)
plt.close()

printwtime("Creating Control .tif")

warped_images = [warp_image(target_full, s, h) for h, s, empty in zip(homographies_full, source_tiles_full, is_empty) if not empty]
warped_combined = np.zeros(list(target_full.shape)+[3])
warped_combined[...,0] = claher(target_full)
for i, img in zip(not_empty_ind, warped_images):
    warped_combined[img!=0,1 + (i%2 + i//Ntilesaxes[0])%2] = img[img!=0]
warped_combined[...,1] = claher(warped_combined[...,1].astype(int))
warped_combined[...,2] = claher(warped_combined[...,2].astype(int))
warped_combined = (np.moveaxis(warped_combined,-1,0)*255).astype("H")
save_tiff(controltifpath, warped_combined)

def in_slice(n, s, length):
    return n in range(*s.indices(length))
def register_counts_tiled(pathin, pathout, homographies, slices, target, source, not_empty_ind, flipsource, verbose=False):
    """ Numpy is (row, column), i.e. (y, x)!!!!!
    """
    counts = pd.read_table(pathin,
                            header=None, names=["x","y","z","Gene","FP"], usecols=list(range(5)))
    if flipsource: counts["y"] = source.shape[0]-1-counts["y"]
    if verbose: print(len(counts),"total counts.")
    
    counts["tile"] = [np.where([in_slice(x, sl[1], source.shape[1]) and in_slice(y, sl[0], source.shape[0]) \
                            for sl in slices])[0][0] for x, y in zip(counts["x"],counts["y"])]
    counts = counts[counts["tile"].apply(lambda x: x in not_empty_ind)]
    counts["x"] -= np.asarray([slices[i][1].start for i in counts["tile"]])
    counts["y"] -= np.asarray([slices[i][0].start for i in counts["tile"]])
    
    counts[["xReg","yReg"]] = [transform_coordinate(homographies[s], x, y) for x, y, s in zip(counts["x"],counts["y"],counts["tile"])]
    
    counts["xReg"] = np.round(counts["xReg"]).astype(int)
    counts["yReg"] = np.round(counts["yReg"]).astype(int)
    counts = counts.loc[np.logical_and(counts["xReg"]>=0, counts["xReg"]<target.shape[1])]
    counts = counts.loc[np.logical_and(counts["yReg"]>=0, counts["yReg"]<target.shape[0])]
    counts = counts[["xReg","yReg","z","Gene","FP"]]
    if verbose: print(len(counts),"registered counts.")
        
    counts.to_csv(pathout, index=False, header=False, sep="\t")

register_counts_tiled(transcriptpath,
                transcriptoutpath,
                homographies_full, slices_full, target_full, source_full, not_empty_ind, flipsource,
                verbose=True)

counts = pd.read_table(transcriptoutpath,
                            header=None, names=["x","y","z","Gene","FP"], usecols=list(range(5)))

fig, ax = plt.subplots(1,1,figsize=(20,20))
ax.imshow(target_full, cmap='gray',origin="lower")
ax.scatter(counts["x"],counts["y"], s=0.1)
plt.savefig(pdftranscriptpath)
plt.close()



