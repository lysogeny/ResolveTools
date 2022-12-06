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
