import numpy as np
import cv2

from ..image.utils import save_tiff, claher, resize_shrink

##############################
### Create tiles
##############################

def tile_2Dimage(image, tile_size):
    """ Tile image and return tiles, slices etc.
    """
    Ntilesaxes = np.asarray(image.shape)//tile_size
    Ntiles = np.prod(Ntilesaxes)
    
    slices = []
    for i in range(Ntiles):
        slices.append((slice((i%Ntilesaxes[0])*tile_size, (i%Ntilesaxes[0]+1)*tile_size, None),
              slice((i//Ntilesaxes[0])*tile_size, (i//Ntilesaxes[0]+1)*tile_size, None)))
    
    image_tiles = [image[sl] for sl in slices]
    
    return image_tiles, slices, Ntiles, Ntilesaxes
