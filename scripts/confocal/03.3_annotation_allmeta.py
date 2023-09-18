import os
from tqdm import tqdm
import re
import pandas as pd
import numpy as np

import sys
sys.path.insert(0,'/data/')
from resolve_tools.segmentation.brainregions import regionkey, regioncolors
from resolve_tools.utils.utils import printwtime

path = "/data/confocal/03_annotation/"
files = list(filter(lambda x: "_annotated_regions.npz" in x, os.listdir("/data/confocal/03_annotation")))

df = pd.DataFrame(index=np.arange(regioncolors.shape[0]))

for file in files:
    printwtime("Reading annotation {}.".format(file))
    roi = re.search(r'R(\d)_W(\d)A(\d)', file).group(0)
    df[roi] = 0
    u, c = np.unique(np.load(path+file)["regions"], return_counts=True)
    df.loc[u, roi] = c

df = df.loc[[i for i in df.index if i in regionkey.keys()]].copy()
df.index = [regionkey[i] for i in df.index]
df.to_csv(path+"Confocal_annotation_overview.csv")
