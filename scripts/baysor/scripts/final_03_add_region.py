import sys
sys.path.insert(0,'/data/')
import os

from resolve_tools.utils.utils import printwtime
from resolve_tools.segmentation.brainregions import add_region_toadata

result = sys.argv[1] # results_N21_wmesmer_combined

resultfolder = "/data/baysor/04_baysor/"+result
#roi = sys.argv[2] #"R2_W0A2"

rois = os.listdir(resultfolder+"/rois/")

for roi in rois:
    printwtime(roi)
    adatafiles = list(filter(lambda x: ".loom" in x, os.listdir(resultfolder+"/rois/"+roi)))
    for adatafile in adatafiles:
        printwtime("   "+adatafile)
        add_region_toadata(resultfolder+"/rois/"+roi+"/"+adatafile,
                           "/data/confocal/03_annotation/Confocal_"+roi+"_annotated_regions.npz")




