import sys
sys.path.insert(0,'/data/')
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from resolve_tools.utils.utils import printwtime
from resolve_tools.analysis.post import read_add_combined
from resolve_tools.segmentation.counts import read_loom
from resolve_tools.analysis.pseudotime import get_human_glmpca_data, get_meanpseudotime, plot_PT_shares

result = sys.argv[1] # results_N21_wmesmer_combined
resultfolder = "/data/baysor/04_baysor/"+result

path = resultfolder+"/QC_pseudotime/"
if not os.path.exists(path): os.makedirs(path)


Nmin = 3
humancutoff=0.4

adata = read_add_combined(resultfolder+"/"+result+"_segmentation_cells.loom", humancutoff=humancutoff)

##### Cell Size Plots
printwtime("Making Cell Size Plots")
fig, ax = plt.subplots(3,1,figsize=(15,10))
ax[0].hist(adata.obs["TotalGeneCount"], bins=np.arange(150)-0.5, histtype="step")
ax[0].set_yscale("log")
ax[1].hist(adata.obs["TotalGeneCount"], bins=np.arange(50)-0.5, histtype="step")
ax[1].set_yscale("log")
for ctype in np.unique(adata.obs["BaysorClusterCelltype"]):
    ax[2].hist(adata.obs.loc[adata.obs["BaysorClusterCelltype"]==ctype, "TotalGeneCount"],
             bins=np.arange(100)-0.5, histtype="step", label=ctype)
ax[2].legend()
ax[2].set_yscale("log")
plt.savefig(path+"cellsize_dist.jpg", dpi=300, bbox_inches="tight")
plt.close()

adata = adata[adata.obs["TotalGeneCount"]>Nmin].copy()

##### Human Share Plots
printwtime("Making Human Share Plots")
fig, ax = plt.subplots(1,1,figsize=(25,10))
sns.violinplot(adata.obs, x="BaysorClusterCelltype", y="HumanGeneShare", width=2, inner=None, ax=ax)
plt.xticks(rotation=45);
ax.hlines(y=humancutoff, xmin=-.5,
          xmax=len(np.unique(adata.obs["BaysorClusterCelltype"]))+.5, color="black")
plt.savefig(path+"humanshare_dist.jpg", dpi=300, bbox_inches="tight")
plt.close()

##### Gene Distribution Plots
printwtime("Making Gene Distribution Plots")
df = adata.to_df()
df.columns = adata.var["GeneClass"]
df.index = adata.obs["BaysorClusterCelltype"]
df = df.groupby("BaysorClusterCelltype").apply(lambda x: x.sum(axis=0))
df = df.groupby("GeneClass", axis=1).apply(lambda x: x.sum(axis=1)).astype(int).T

dfp = np.round(df/np.asarray(df.sum(axis=1))[:,None]*100,0).astype(int)#.astype(str).replace("0","")
sns.heatmap(dfp, annot=True, cbar=False).get_figure().savefig(path+"geneshare_clusters_genenorm.jpg", dpi=300, bbox_inches="tight")
plt.close()

dfp = np.round(df/np.asarray(df.sum(axis=0))[None]*100,0).iloc[:,:-1].astype(int)#.astype(str).replace("0","")
sns.heatmap(dfp, annot=True, cbar=False).get_figure().savefig(path+"geneshare_clusters_clusternorm.jpg", dpi=300, bbox_inches="tight")
plt.close()

##### Add var

adatabay = read_loom(resultfolder+"/"+result+"_baysor_cells.loom")
adata.var["TotalNoNoise"] = adatabay.to_df().sum(axis=0).astype(int)
adata.var["TotalInSeg"] = adata.to_df().sum(axis=0).astype(int)
adata.var["TotalInSegShare"] = adata.var["TotalInSeg"]/adata.var["TotalNoNoise"]

##### Pseudotime
#printwtime("Finding Pseudotime")
#adatapca, countsshort, adatapcashort = get_human_glmpca_data(adata)
#pt, pts, inds = get_meanpseudotime(countsshort, adatapcashort, meanN=10, N=2,
#                                   verbose=False, separation=0.3, tryFull=False)

#printwtime("Making Pseudotime Plots")
#plot_PT_shares(pt, adatapcashort, hue="BaysorClusterCelltype")
#plt.savefig(path+"rankpt_genegroups_celltype.jpeg", dpi=400)
#plt.close()
#plot_PT_shares(pt, adatapcashort, hue="ROIShort")
#plt.savefig(path+"rankpt_genegroups_ROIShort.jpeg", dpi=400)
#plt.close()

#printwtime("Saving PT")
#adata.obs["rankPT"] = np.nan
#adata.obs.loc[adatapcashort.obs.index, "rankPT"] = pt
adata.write_loom(resultfolder+"/"+result+"_segmentation_cells_QC.loom")








