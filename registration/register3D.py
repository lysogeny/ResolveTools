import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

##############################
### Resolve Counts
### 
### 
##############################

def get_tiled_mean_counts(counts, binsize = 1000):
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
            means[i,j] = counts.loc[np.logical_and(maskx, masky),"z"].mean()
            xs[i,j] = counts.loc[np.logical_and(maskx, masky),"x"].mean() # (j+0.5)*binsize
            ys[i,j] = counts.loc[np.logical_and(maskx, masky),"y"].mean() # (i+0.5)*binsize
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

def fit_plane(xs, ys, vals_, p0=[5e-4, 5e-4, 4]):
    """ Fit plane to points.
    """
    vals = vals_.flatten()
    x = np.asarray([xs.flatten(), ys.flatten()])
    popt, pcov = curve_fit(plane, x, vals, p0=p0)
    return popt, pcov

def point_to_new_plane(x, y, z, target, source):
    """ Take (x,y,z) in source plane, transform z to target plane.
        Only appropriate if relative angles are very small!
    """
    znew = plane([x,y], *(target-source)) + z
    return znew

def counts_to_plane(counts, target, source):
    """ Transform counts into new plane, shifting z with point_to_new_plane.
    """
    counts["z"] = [point_to_new_plane(x, y, z, target, source) for x, y, z in zip(counts["x"], counts["y"], counts["z"])]