import sys
sys.path.insert(0,'/data/')
import os
import matplotlib.pyplot as plt

from resolve_tools.baysor.visualization import plot_final_assignment_post
from resolve_tools.utils.utils import printwtime

## Split Results

result = sys.argv[1] # results_N21_wmesmer_combined
key = sys.argv[2] # _wmesmer_combined_combinekey

resultfolder = "/data/baysor/04_baysor/"+result
keyfile = "/data/baysor/03_transcripts_combined/"+key+".npz"
genemetafile = "/data/metadata/gbm_resolve_genes.csv"
backgroundtemplate = "/data/confocal/01_image/Confocal_{}_DAPI_max_claher.tif"

plot_final_assignment_post(resultfolder, keyfile, genemetafile, backgroundtemplate=backgroundtemplate)
