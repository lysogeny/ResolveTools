import numpy as np
import pandas as pd

from ..segmentation.counts import read_loom

##############################
### Loading
##############################

def read_add_combined(path, humancutoff=0.4):
    """ Read combined and processed baysor results, add some metadata etc.
    """
    adata = read_loom(path)

    adata.var["GeneClass"] = adata.var["Species"] + " - " + adata.var["Celltype"]

    counts = adata.to_df()
    adata.obs["TotalGeneCount"] = counts.sum(axis=1)
    adata.obs["MouseGeneCount"] = counts[adata.var.loc[adata.var["Species"]=="Mouse"].index].sum(axis=1)
    adata.obs["HumanGeneCount"] = counts[adata.var.loc[adata.var["Species"].apply(lambda x: "Human" in x)].index].sum(axis=1)
    adata.obs["MouseGeneShare"] = adata.obs["MouseGeneCount"]/counts.sum(axis=1)
    adata.obs["HumanGeneShare"] = adata.obs["HumanGeneCount"]/counts.sum(axis=1)

    def shorten(x):
        for ops in ["CC ","CTX ","LV ","SN ","STR ","TM "]:
            if x==ops[:-1] or ops in x:
                return ops[:-1]
        return x
    adata.obs["BrainRegionNameShort"] = adata.obs["BrainRegionName"].apply(shorten)

    def shorten(x):
        run = int(x[1])
        sl = int(x[4])
        if run == 1:
            if sl<4: return "reporter"
            else: return "naive"
        elif run == 2:
            if sl<4: return "sFRP1"
            elif sl<6: return "naive"
            else: return "reporter"
        else:
            return x
    adata.obs["ROIShort"] = adata.obs["ROI"].apply(shorten)

    adata.obs["IS_HUMAN"] = adata.obs["HumanGeneShare"]>humancutoff
    
    return adata