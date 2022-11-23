import numpy as np

##############################
### Brain Regions Initial
##############################

regionkey = {1 : ("cortex", [255,0,0]), # 3
             2 : ("striatum", [182,255,0]), # 6
             3 : ("coprus callosum", [0,38,255]), # 12
             4 : ("lateral ventricle", [255,106,0]), # 4
             5 : ("septal nuclei", [255,216,0]), # 5
             6 : ("tumor", [178,0,255]), # 14
             }

def processes_regionsegmentation_initial(seg, regkey = regionkey):
    """ Processes initial color region segmentation with key, return region mask.
        Could use some slight postprocessing, but probably not important.
    """
    segregions = np.zeros(list(seg.shape)[:-1])
    for key in regkey.keys():
        segregions[np.all(seg[...,::-1]==regkey[key][1], axis=-1)] = key
    return segregions