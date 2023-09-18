import sys
sys.path.insert(0,'/data/')
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from resolve_tools.baysor.utils import split_baysor_ROIs, assign_counts_from_Baysor
from resolve_tools.baysor.visualization import plot_celltypedist
from resolve_tools.utils.utils import printwtime

## Split Results

result = sys.argv[1] # results_N21_wmesmer_combined
key = sys.argv[2] # _wmesmer_combined_combinekey

printwtime("Split Results")
keyfile = "/data/baysor/03_transcripts_combined/"+key+".npz"
idfile = "/data/baysor/04_baysor/"+result+"/cluster_ids.npz"
genemetafile = "/data/metadata/gbm_resolve_genes.csv"

resultfolder = "/data/baysor/04_baysor/"+result
ignore_indices = [9]

split_baysor_ROIs(resultfolder, keyfile, idfile, genemetafile, ignore_indices=ignore_indices)


## Create Baysor Loom

printwtime("Create Baysor .loom")
rois = os.listdir(resultfolder+"/rois/")

for roi in rois:
    #print(roi)
    adata = assign_counts_from_Baysor(resultfolder+"/rois/"+roi, genemetafile, roi, clusteridfile=idfile)
    adata.write_loom(resultfolder+"/rois/"+roi+"/baysor_cells.loom")


## Plot Baysor Celltypes

printwtime("Plot Baysor celltypes")
path = resultfolder+"/baysor_plots/"
if not os.path.exists(path): os.makedirs(path)
rois = os.listdir(resultfolder+"/rois/")

for roi in rois:
    #print(roi)    
    plot_celltypedist(resultfolder+"/rois/"+roi+"/baysor_cells.loom",
                      "/data/confocal/02_mesmer_nuclei/Confocal_"+roi+"_DAPI_mesmer_nuclei_post_meta.csv",
                      idfile,
                      outfile=path+roi+"_baysor_celltypes.png",
                      title = roi)

