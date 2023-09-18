import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
import re
import matplotlib.pyplot as plt

from ..resolve.resolveimage import read_Resolve_count
from ..image.utils import read_single_modality_confocal
from ..resolve.resolveimage import RESOLVE_VOXEL_SIZE
from ..utils.utils import printwtime

##############################
### Resolve Counts
##############################

def get_tiled_mean_counts(counts, binsize = 1000, Ncutoff = 20):
    """ Get mean z position across Resolve counts in tiles of size binsize x binsize.
    """
    Nx, Ny = counts["x"].max()//binsize, counts["y"].max()//binsize
    means = np.zeros((Ny, Nx))
    xs = np.zeros((Ny, Nx))
    ys = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            maskx = np.logical_and(counts["x"]>=j*binsize, counts["x"]<(j+1)*binsize)
            masky = np.logical_and(counts["y"]>=i*binsize, counts["y"]<(i+1)*binsize)
            mask = np.logical_and(maskx, masky)
            means[i,j] = counts.loc[mask,"z"].mean() if mask.sum()>Ncutoff else np.nan
            xs[i,j] = counts.loc[mask,"x"].mean() if mask.sum()>Ncutoff else np.nan # (j+0.5)*binsize
            ys[i,j] = counts.loc[mask,"y"].mean() if mask.sum()>Ncutoff else np.nan # (i+0.5)*binsize
    return means, xs, ys

def get_tiled_mean_counts_fromfile(file, binsize = 1000):
    """ get_tiled_mean_counts from file of 2D registered counts.
    """
    counts = pd.read_table(file,
                            header=None, names=["x","y","z","Gene","FP"], usecols=list(range(5)))
    counts["z"] = counts["z"]*0.3125 # Hardcoded resolution from Resolve
    return get_tiled_mean_counts(counts)

##############################
### Confocal Image
##############################

def get_tiled_mean_image(img, binsize = 1000, mode="mean"):
    """ Get either mean or max position along z axis of image,
        in tiles of size binsize x binsize.
        Assumes 1um distance of the z stacks.
    """
    Nx, Ny = img.shape[2]//binsize, img.shape[1]//binsize
    means = np.zeros((Ny, Nx))
    xs = np.zeros((Ny, Nx))
    ys = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            tile = img[:, i*binsize:(i+1)*binsize, j*binsize:(j+1)*binsize]
            ztotal = tile.sum(axis=-1).sum(axis=-1)
            ztotal = ztotal/ztotal.sum()
            if mode=="mean":
                zmean = (ztotal*np.arange(ztotal.shape[0])).sum()
                means[i,j] = zmean
            if mode=="max":
                zmax = np.argmax(ztotal)
                means[i,j] = zmax
            xs[i,j] = (j+0.5)*binsize
            ys[i,j] = (i+0.5)*binsize
    return means, xs, ys

def get_tiled_mean_images(imgs, binsize = 1000, mode="mean"):
    """ get_tiled_mean_image, with mean across images.
    """
    #means = [get_tiled_mean_image(img, binsize, mode) for img in imgs]
    #return np.asarray(means).mean(axis=0)
    pass

##############################
### Planes
##############################

def plane(x, a, b, c):
    """ Plane, x[0] should be x and x[1] y.
    """
    return x[0]*a + x[1]*b + c

def fit_plane(vals_, xs, ys, p0=[5e-4, 5e-4, 4]):
    """ Fit plane to points.
    """
    vals = vals_.flatten()
    x = np.asarray([xs.flatten(), ys.flatten()])
    popt, pcov = curve_fit(plane, x[:,~np.isnan(vals)], vals[~np.isnan(vals)], p0=p0)
    return popt, pcov

def point_to_new_plane(x, y, z, target, source):
    """ Take (x,y,z) in source plane, transform z to target plane.
        Only appropriate if relative angles are very small!
    """
    znew = plane([x,y], *(target-source)) + z
    return znew

##############################
### Register Counts
##############################

def counts_to_plane(counts, target, source):
    """ Transform counts into new plane, shifting z with point_to_new_plane.
    """
    counts["z"] = [point_to_new_plane(x, y, z, target, source) for x, y, z in zip(counts["x"], counts["y"], counts["z"])]

def register_3d_counts(countfile, dapifile, outfile, segmentationfile="", segmentationkey="mask_post",
                       verbose=True, binsize=1000, plotfile="", plottitle="", shiftstep=0.4):
    """ Register Resolve 3D counts to 3D confocal DAPI image.
    """
    if verbose: printwtime("Loading Image")
    dapi = read_single_modality_confocal(dapifile)
    imageshape = dapi.shape
    
    countplane = fit_plane(*get_tiled_mean_counts_fromfile(countfile, binsize))[0]
    dapiplane = fit_plane(*get_tiled_mean_image(dapi, binsize, mode="mean"))[0]
    del dapi
    
    counts = read_Resolve_count(countfile)
    if verbose: printwtime(f"Initially {len(counts)} counts")
    counts["z"] = counts["z"]*RESOLVE_VOXEL_SIZE[0] # Hardcoded resolution from Resolve
    counts_to_plane(counts, dapiplane, countplane)
    
    if segmentationfile:
        if verbose: printwtime("Loading Segmentation")
        segmentation = np.load(segmentationfile)[segmentationkey]

        def shift_counts(counts_, shift):
            counts = counts_.copy()
            counts["z"] += shift
            counts["z"] = np.round(counts["z"],0).astype(int)
            counts = counts[np.logical_and(counts["z"]>0, counts["z"]<imageshape[0])]
            return counts
        def get_share(segmentation, counts, shifts=[]):
            shares = []
            total = len(counts)
            for shift in shifts:
                shifted = shift_counts(counts, shift)
                shares.append((segmentation[shifted["z"], shifted["y"], shifted["x"]]!=0).sum()/total)
            return shares

        shifts = np.arange(-8,6,shiftstep)
        shares = get_share(segmentation, counts, shifts)
        del segmentation
        final_shift = shifts[np.argmax(shares)]
        if verbose: printwtime(f"Found additional z shift of {np.round(final_shift,1)} um")

        if plotfile:
            plt.plot(shifts, shares, color="red")
            plt.xlabel("z Shift [um]")
            plt.ylabel("Share of Transcripts in Segmentation")
            plt.vlines(x=final_shift, ymin=min(shares), ymax=max(shares), color="black")
            plt.ylim([min(shares), None])
            if plottitle: plt.title(plottitle)
            plt.savefig(plotfile)
            plt.close()
    else:
        printwtime("No segmentation was given, so no additional z shift can be approximated")
        finalshift = 0
    
    counts = shift_counts(counts, final_shift)
    if verbose: printwtime(f"{len(counts)} counts after 3D registration")
    
    counts.to_csv(outfile, index=False, header=False, sep="\t")
