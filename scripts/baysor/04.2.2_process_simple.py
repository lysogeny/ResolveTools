import sys
sys.path.insert(0,'/data/')
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ResolveTools.baysor.utils import split_baysor_ROIs, assign_counts_from_Baysor
from ResolveTools.baysor.visualization import plot_celltypedist
from ResolveTools.utils.utils import printwtime

## Split Results

printwtime("Split Results")
keyfile = "/data/baysor/03_transcripts_combined/T6GBM_transcripts_combined_combinekey.npz"
idfile = "/data/baysor/04_baysor/reference_N21/cluster_ids.npz"
genemetafile = "/data/metadata/gbm_resolve_genes.csv"

resultfolder = "/data/baysor/04_baysor/results_N21_combined"
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
                      outfile=path+roi+"_baysor_celltypes.pdf",
                      title = roi)

    plot_celltypedist(resultfolder+"/rois/"+roi+"/baysor_cells.loom",
                      "/data/confocal/02_mesmer_nuclei/Confocal_"+roi+"_DAPI_mesmer_nuclei_post_meta.csv",
                      idfile,
                      outfile=path+roi+"_baysor_celltypes.png",
                      title = roi)

#with PdfPages(path+'AllRegions_baysor_celltypes.pdf') as pdf:
#    for roi in rois:
#        #print(roi)
#        plot_celltypedist(resultfolder+"/rois/"+roi+"/baysor_cells.loom",
#                          "/data/confocal/02_mesmer_nuclei/Confocal_"+roi+"_DAPI_mesmer_nuclei_post_meta.csv",
#                          idfile,
#                          title = roi)
#        pdf.savefig()
#        plt.close()


