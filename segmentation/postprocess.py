import numpy as np
from scipy.ndimage import center_of_mass
from numba import njit
from tqdm import tqdm
from datetime import datetime

##############################
### Mask Helpers
##############################

def _remove_labels(mask, labels):
    """ Remove labels from mask.
    """
    repl = np.arange(0,mask.max()+1,1,int)
    repl[labels] = 0
    return repl[mask]

def _only_keep_labels(mask, labels):
    """ Only keep labels, drop all else from mask.
    """
    repl = np.zeros((mask.max()),int)
    repl[labels] = labels
    return repl[mask]

def _relabel_from_zero(mask):
    """ Relabel cells with consecutive numbers.
    """
    repl = np.arange(0,mask.max()+1,1, int)
    u = np.unique(mask)
    repl[u] = np.arange(u.shape[0])
    return repl[mask]

##############################
### Mask Intersection
### 
### Mostly copied form mesmer
##############################

@njit
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    # put label arrays into standard form then flatten them 
#     x = (utils.format_labels(x)).ravel()
#     y = (utils.format_labels(y)).ravel()
    x = x.ravel()#.astype(int)
    y = y.ravel()#.astype(int)
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((int(1+x.max()),int(1+y.max())), dtype=np.uint)
    #overlap = np.zeros((4, 4), dtype=np.uint)
    
    for i in range(len(x)):
        overlap[int(x[i]),int(y[i])] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]
    
    ------------
    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken 
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix. 

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _intersection_over_first(masks_true, masks_pred):
    """ intersection over first mask
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_true)
    iou[np.isnan(iou)] = 0.0
    return iou

##############################
### 3D Mask Stitching
##############################

def stitch3D_initial(masks_, stitch_threshold=0.25, verbose=False):
    """ Stitch 2D masks into 3D volume with stitch_threshold on IOU
        Start at zero, since it doesn't matter here.
    """
    masks = masks_.copy() # Copy, in-place has weird stuff!
    mmax = masks[0].max() # max label number of current mask
    empty = 0
    
    rpt = tqdm if verbose else lambda x:x
    for i in rpt(range(len(masks)-1)):
        iou = _intersection_over_union(masks[i+1], masks[i])[1:,1:]
        if not iou.size and empty == 0: # One mask has no labels, and none of the previous ones did
            masks[i+1] = masks[i+1] # Why?? Just for symmetry of the code?
            mmax = masks[i+1].max()
        elif not iou.size and not empty == 0: # One mask has no labels, and some of the previous ones did
            icount = masks[i+1].max()
            istitch = np.arange(mmax+1, mmax + icount+1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i+1] = istitch[masks[i+1]] # Change cell indices to conform to new unique naming
        else:
            iou[iou < stitch_threshold] = 0.0 # Only above threshold
            iou[iou < iou.max(axis=0)] = 0.0 # For low threshold, keep the one with highest intersection with mask i+1
            istitch = iou.argmax(axis=1) + 1 # Index of corresponding cell in i to cell in i+1
            ino = np.nonzero(iou.max(axis=1)==0.0)[0] # Don't stitch those
            istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int) # Instead raise cell index for those
            mmax += len(ino) # Increase overall total
            istitch = np.append(np.array(0), istitch) # replace 0 with 0
            masks[i+1] = istitch[masks[i+1]] # Stitch mask i+1
            empty = 1
            
    return masks

def _find_layer_disconnected(masks,i):
    """ Find cells that don't have any connection in adjacent layers.
    """
    unique_i = set(np.unique(masks[i]))
    if i == 0: unique = set(np.unique(masks[i+1]))
    elif i == len(masks)-1: unique = set(np.unique(masks[i-1]))
    else: unique = set(np.unique(masks[i-1])) | set(np.unique(masks[i+1]))
    return np.asarray(list(unique_i - unique)).astype(int)

def _stitch_layer_disconnected(masks, i, intersection_cutoff=0.8):
    """ Stitch disconnected parts in single layer.
    """
    disconnected = _find_layer_disconnected(masks,i)
    if len(disconnected)==0: return []
    if i==0 or i==len(masks)-1:
        iof = _intersection_over_first(masks[i], masks[i+1 if i==0 else i-1])[1:,1:]
        if np.prod(iof.shape)==0: # The closest mask doesn't have any cells
            return list()
        maxlabels = iof.argmax(axis=1)[disconnected-1]+1
        inters = iof[disconnected-1,maxlabels-1]
    else:
        iof_up = _intersection_over_first(masks[i], masks[i+1])[1:,1:]
        iof_down = _intersection_over_first(masks[i], masks[i-1])[1:,1:]
        
        if np.prod(iof_up.shape)==0 and np.prod(iof_down.shape)==0: # Both sides are empty
            return list()
        elif np.prod(iof_up.shape)==0: # Only up is empty
            maxlabels = iof_down.argmax(axis=1)[disconnected-1]+1
            inters = iof_down[disconnected-1,maxlabels-1]
        elif np.prod(iof_down.shape)==0: # Only down is empty
            maxlabels = iof_up.argmax(axis=1)[disconnected-1]+1
            inters = iof_up[disconnected-1,maxlabels-1]
        else: # Both not empty
            maxlabels_up = iof_up.argmax(axis=1)[disconnected-1]+1
            inters_up = iof_up[disconnected-1,maxlabels_up-1]
            maxlabels_down = iof_down.argmax(axis=1)[disconnected-1]+1
            inters_down = iof_down[disconnected-1,maxlabels_down-1]
            
            udind = np.asarray([inters_up,inters_down]).argmax(axis=0)
            maxlabels = np.asarray([maxlabels_up,maxlabels_down])[udind,list(range(len(inters_up)))]
            inters = np.asarray([inters_up,inters_down])[udind,list(range(len(inters_up)))]
    
    if (inters>intersection_cutoff).sum()==0: # Nothing to replace
        return list()
    
    replabels = np.arange(0,masks[i].max()+1,1, int)
    replabels[disconnected[inters>intersection_cutoff]] = maxlabels[inters>intersection_cutoff]
    masks[i] = replabels[masks[i]]
    return list(disconnected[inters>intersection_cutoff])

def stitch3D_stitch_disconnected(masks_, intersection_cutoff=0.6, verbose=False, inplace=False):
    """ Stitch disconnected labels across the whole mask, start in middle.
    """
    masks = masks_ if inplace else masks_.copy()
    stitched = []
    rpt = tqdm if verbose else lambda x:x
    for i in rpt(range(masks.shape[0]//2)[::-1]): # Down from the middle
        stitched += _stitch_layer_disconnected(masks, i, intersection_cutoff)
    for i in rpt(range(masks.shape[0]//2,masks.shape[0])): # Up from middle
        stitched += _stitch_layer_disconnected(masks, i, intersection_cutoff)
    return masks, stitched

def stitch3D_drop_small(masks_, vol_cutoff=800):
    """ Drop all labels with volume below vol_cutoff.
        17/(0.14*0.14*1)\approx 850 um³, repeat with real voxel size and stitch to other nuclei!!!!!!!!!
        Combine this with relabel!
    """
    masks = masks_#.copy()
    u, c = np.unique(masks, return_counts=True)
    drop = u[c<vol_cutoff]
    return _remove_labels(masks, drop), list(drop)

def postprocess_raw_mesmer_masks(masks_, stitch_threshold=0.3, intersection_cutoff=0.6, vol_cutoff=850, verbose=True, verbosetqdm=False):
    """ Postprocess raw mesmer mask.
    """
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"- Starting 3D stitching")
    masks = stitch3D_initial(masks_, stitch_threshold=stitch_threshold, verbose=verbosetqdm)
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"-",np.unique(masks).shape[0],"cells after initial stitching.")
    
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"- Stitching disconnected")
    masks, _ = stitch3D_stitch_disconnected(masks, intersection_cutoff=intersection_cutoff, verbose=verbosetqdm)
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"-",np.unique(masks).shape[0],"cells after stitching disconnected.")
    
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"- Dropping Small cells")
    masks, _ = stitch3D_drop_small(masks, vol_cutoff=vol_cutoff)
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"-",np.unique(masks).shape[0],"cells after dropping unreasonably small cells. IMRPOVE THIS!")
    
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"- Relabel cells")
    masks = _relabel_from_zero(masks)
    
    return masks




