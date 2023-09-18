import sys
sys.path.insert(0,'/data/')
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from resolve_tools.baysor.roi import apply_combine_baysor_output
from resolve_tools.baysor.visualization import plot_celltypedist
from resolve_tools.utils.utils import printwtime

result = sys.argv[1] # results_N21_wmesmer_combined

resultfolder = "/data/baysor/04_baysor/"+result
#roi = sys.argv[2] #"R2_W0A2"

rois = os.listdir(resultfolder+"/rois/")

plotpath = resultfolder+"/custom_assignment_plots/"
if not os.path.exists(plotpath): os.makedirs(plotpath)

for roi in rois:
    idfile = "/data/baysor/04_baysor/"+result+"/cluster_ids.npz"
    genemetafile = "/data/metadata/gbm_resolve_genes.csv"
    
    resultfolderroi = resultfolder+"/rois/"+roi
    background = "/data/confocal/01_image/Confocal_"+roi+"_DAPI_max_claher.tif"
    segloomfile = "/data/resolve/05_counts_naive/T6GBM_"+roi+"_counts_naive.loom"
    boundaryfile = "/data/confocal/02_mesmer_nuclei/Confocal_"+roi+"_DAPI_mesmer_nuclei_post_boundary.npz"
    segmetafile = "/data/confocal/02_mesmer_nuclei/Confocal_"+roi+"_DAPI_mesmer_nuclei_post_meta.csv"
    
    printwtime("- - - - - Processing ROI {} of run {}".format(roi, resultfolder))
    
    apply_combine_baysor_output(resultfolderroi, segloomfile, genemetafile, boundaryfile, idfile, background=background,
                                plotpath=plotpath+roi+"_custom_cell_assignment.jpeg", plotwbackpath=plotpath+roi+"_custom_cell_assignment_wbackground.jpeg")

    plot_celltypedist(resultfolder+"/rois/"+roi+"/segmentation_cells.loom",
                      segmetafile,
                      idfile,
                      outfile=plotpath+roi+"_baysor_celltypes_acustom.png",
                      title = roi)





