import os
import re
import sys
sys.path.insert(0,'/data/')
from resolve_tools.segmentation.counts import assign_counts_from_segmentation
from resolve_tools.utils.utils import printwtime

path = "/data/resolve/04_registration3D/"

names = [re.search(r'R(\d)_W(\d)A(\d)', filename).group(0) for filename in os.listdir(path)]

for name in names:
    printwtime(name)
    assign_counts_from_segmentation("/data/resolve/04_registration3D/T6GBM_" + name + "_transcripts_registered3D.txt",
                                    "/data/metadata/gbm_resolve_genes.csv",
                                    "/data/confocal/02_mesmer_nuclei/Confocal_" + name + "_DAPI_mesmer_nuclei_post.npz",
                                    "mask_post",
                                    "/data/confocal/02_mesmer_nuclei/Confocal_" + name + "_DAPI_mesmer_nuclei_post_meta.csv",
                                    "/data/resolve/05_counts_naive/T6GBM_" + name + "_counts_naive.loom",
                                    name,
                                    True)
