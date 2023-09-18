import sys
sys.path.insert(0,'/data/')
import os

from resolve_tools.utils.utils import printwtime
from resolve_tools.baysor.utils import add_stain_to_ROI

result = sys.argv[1] # results_N21_wmesmer_combined

resultfolder = "/data/baysor/04_baysor/"+result
imagefolder = "/data/confocal/01_image"
segmentationfolder = "/data/confocal/02_mesmer_nuclei"

rois = os.listdir(resultfolder+"/rois/")

for roi in rois:
    add_stain_to_ROI(resultfolder, imagefolder, segmentationfolder, roi, extend=[1,10,10])




