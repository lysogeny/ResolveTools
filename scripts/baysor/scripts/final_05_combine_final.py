import sys
sys.path.insert(0,'/data/')
import os

from resolve_tools.utils.utils import printwtime
from resolve_tools.baysor.utils import combine_adatas

result = sys.argv[1] # results_N21_wmesmer_combined

resultfolder = "/data/baysor/04_baysor/"+result
genemetafile = "/data/metadata/gbm_resolve_genes.csv"

combine_adatas(resultfolder, genemetafile,
               loomfile = "segmentation_cells.loom",
               outfile = result+"_segmentation_cells.loom")
combine_adatas(resultfolder, genemetafile,
               loomfile = "baysor_cells_post.loom",
               outfile = result+"_baysor_cells.loom")




