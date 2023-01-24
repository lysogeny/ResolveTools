import numpy as np

##############################
### Brain Regions Initial
##############################

from .visualize import get_rgb_distinct
from matplotlib.colors import rgb2hex

regionkey = {   1 : "CTX", #
                2 : "CC", #
                3 : "SN", #
                4 : "LV", #
                5 : "STR", #
                
                6 : "CTX CLH", #
                7 : "CC CLH", #
                8 : "SN CLH", #
                9 : "LV CLH", #
               11 : "STR CLH", #
               
               12 : "TM IS", #
               13 : "TM", #
               14 : "", #
               15 : "", #
               16 : "" #
             }

#(get_rgb_distinct(30)*255).astype(int)
regioncolors = np.asarray([[  0,   0,   0],
                           [ 74, 161, 255],
                           [ 65,  78, 255],
                           [ 82, 255, 165],
                           [ 14, 255,  80],
                           [160, 255, 111],
                           [218, 255, 105],
                           [ 35, 186, 255],
                           [ 67, 255,  80],
                           [223, 125, 255],
                           [255, 164,  99],
                           [ 39, 255, 232],
                           [255,  72,  72],
                           [ 57, 111, 255],
                           [  5, 229, 255],
                           [ 12, 255, 179],
                           [255,  94,  52],
                           [247,  51, 255],
                           [255, 120, 176],
                           [255,  24,  72],
                           [123,  54, 255],
                           [173,  73, 255],
                           [255,  60, 181],
                           [ 99, 255,  74],
                           [247, 255,  34],
                           [ 40,   6, 255],
                           [255,  23,  23],
                           [255,  82, 225],
                           [159, 255,  42],
                           [255, 181,  61],
                           [255, 221,  60]])

#"_nl_".join([rgb2hex(x).replace("#","ff") for x in regioncolors/255])
#ff000000_nl_ff4aa1ff_nl_ff414eff_nl_ff52ffa5_nl_ff0eff50_nl_ffa0ff6f_nl_ffdaff69_nl_ff23baff_nl_ff43ff50_nl_ffdf7dff_nl_ffffa463_nl_ff27ffe8_nl_ffff4848_nl_ff396fff_nl_ff05e5ff_nl_ff0cffb3_nl_ffff5e34_nl_fff733ff_nl_ffff78b0_nl_ffff1848_nl_ff7b36ff_nl_ffad49ff_nl_ffff3cb5_nl_ff63ff4a_nl_fff7ff22_nl_ff2806ff_nl_ffff1717_nl_ffff52e1_nl_ff9fff2a_nl_ffffb53d_nl_ffffdd3c


def processes_regionsegmentation_initial(seg, regkey = regionkey):
    """ Processes initial color region segmentation with key, return region mask.
        Could use some slight postprocessing, but probably not important.
    """
    segregions = np.zeros(list(seg.shape)[:-1])
    for key in regkey.keys():
        segregions[np.all(seg[...,::-1]==regkey[key][1], axis=-1)] = key
    return segregions

##############################
### Brain Regions Of Resolve Counts
##############################

def add_regions_to_resolve(pathin, pathout, pathregion, verbose=True):
    """ Add region column to registered resolve transcript list.
    """
    rim = ResolveImage(pathin)
    regions = np.load(pathregion)["regions"]
    rim.full_data["Region"] = regions[rim.full_data["y"], rim.full_data["x"]].astype(int)
    rim.full_data.to_csv(pathout, index=False)
    if verbose: print(np.unique(rim.full_data["Region"], return_counts=True))
