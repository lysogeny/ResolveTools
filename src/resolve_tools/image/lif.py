from readlif.reader import LifFile

##############################
### Reading confocal .lif images
##############################

def read_lif_image(lif, z, channel):
    """ lif.get_frame apparently uses random ordering, but this works.
    """
    return lif._get_item(z*lif.channels + channel)