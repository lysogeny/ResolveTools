import sys
sys.path.insert(0,'/data/')
import os

from ResolveTools.utils.utils import printwtime
from ResolveTools.baysor.utils import combine_adatas

resultfolder = "/data/baysor/04_baysor/results_N21_wmesmer_combined"
genemetafile = "/data/metadata/gbm_resolve_genes.csv"

combine_adatas(resultfolder, genemetafile,
               loomfile = "segmentation_cells.loom",
               outfile = "results_N21_wmesmer_combined_segmentation_cells.loom")
combine_adatas(resultfolder, genemetafile,
               loomfile = "baysor_cells_post.loom",
               outfile = "results_N21_wmesmer_combined_baysor_cells.loom")




