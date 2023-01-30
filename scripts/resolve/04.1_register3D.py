import sys
sys.path.insert(0,'/data/')
from ResolveTools.registration.register3D import register_3d_counts
from ResolveTools.utils.utils import printwtime

import re
import os
files = os.listdir("/data/resolve/03_registration2D/")
donefiles = os.listdir("/data/resolve/04_registration3D/")
names = [re.search(r'R(\d)_W(\d)A(\d)', s).group(0) for s in files if "transcripts_registered2D" in s]
donenames = [re.search(r'R(\d)_W(\d)A(\d)', s).group(0) for s in donefiles if "transcripts_registered3D" in s]
names = [name for name in names if not name in donenames]

for name in names:
    printwtime(name)
    register_3d_counts("/data/resolve/03_registration2D/T6GBM_"+name+"_transcripts_registered2D.txt",
                       "/data/confocal/01_image/Confocal_"+name+"_DAPI.tif",
                       "/data/resolve/04_registration3D/T6GBM_"+name+"_transcripts_registered3D.txt",
                       binsize=1000)