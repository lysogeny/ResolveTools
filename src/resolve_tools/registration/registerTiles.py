import numpy as np

##############################
# Create tiles
##############################


def tile_2Dimage(image, tile_size):
    """
    Tile image and return tiles, slices etc.
    """
    n_tiles_axes = np.asarray(image.shape)//tile_size
    n_tiles = np.prod(n_tiles_axes)

    slices = []
    for i in range(n_tiles):
        slices.append((slice((i % n_tiles_axes[0])*tile_size,
                             (i % n_tiles_axes[0]+1)*tile_size,
                             None),
                       slice((i//n_tiles_axes[0])*tile_size,
                             (i//n_tiles_axes[0]+1)*tile_size,
                             None)))

    image_tiles = [image[sl] for sl in slices]

    return image_tiles, slices, n_tiles, n_tiles_axes
