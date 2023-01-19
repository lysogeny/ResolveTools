import numpy as np
import pandas as pd
import anndata
import re

from ..resolve.resolveimage import ResolveImage
from ..utils.parameters import CONFOCAL_VOXEL_SIZE

##############################
### Translate transcripts between Baysor and Resolve
##############################

def counts_resolve_to_baysor(resolvepath, baysorpath, dropgenes=[], sampling=CONFOCAL_VOXEL_SIZE[:3],
                             segmaskpath="", segmaskkey="mask_post"):
    """ Transform counts from resolve format to Baysor format.
        Can add cell identity from segmentation mask.
    """
    res = ResolveImage(resolvepath)
    if segmaskpath:
        mask = np.load(segmaskpath)[segmaskkey]
        res.full_data["cell"] = mask[res.full_data["z"], res.full_data["y"], res.full_data["x"]]
        #res.full_data["cell"] = res.full_data["cell"].astype(str)
        #res.full_data.loc[res.full_data["cell"]=="0", "cell"] = ""
    res.full_data[["z","y","x"]] = np.round(res.full_data[["z","y","x"]]*sampling,3)
    #res.full_data = res.full_data[["x","y","z","GeneR"]].copy()
    res.full_data = res.full_data.rename(columns={"GeneR":"gene"})
    if len(dropgenes)>0:
        res.full_data = res.full_data[[g not in dropgenes for g in res.full_data["gene"]]]
    res.full_data.to_csv(baysorpath, index=False)

##############################
### Assign counts using only Baysor
##############################

def assign_counts_from_Baysor(resultsfolder, genemetafile, roikey, do_for="cell"):
    """ Assign counts to cells, using Baysor output.
    """
    if not do_for in ["cell", "cluster"]: raise ValueError("Not available")
    
    segmentation = pd.read_table(resultsfolder+"/segmentation.csv", sep=",")
    segmentation = segmentation[~segmentation["is_noise"]]
    
    var = pd.DataFrame(np.unique(segmentation["gene"]))
    var.columns = ["GeneR"]
    var.index = np.asarray(var["GeneR"])
    obs = pd.DataFrame(np.arange(segmentation[do_for].max())+1, columns=["MaskIndex"])
    obs["CellName"] = roikey+"_"+obs["MaskIndex"].astype(str)
    obs.index = np.asarray(obs["MaskIndex"])
    obs["ROI"] = roikey
    
    counts = pd.DataFrame(np.zeros((obs.shape[0],var.shape[0])),
                          columns = var.index,
                          index = obs.index)
    def count_gene(gene, segmentation, counts):
        cell = segmentation.loc[segmentation["gene"]==gene,do_for]
        u, c = np.unique(cell[cell!=0], return_counts=True)
        counts.loc[u,gene] = c
    for gene in var["GeneR"]:
        count_gene(gene, segmentation, counts)
    
    obs.index = np.asarray(obs["CellName"])
    counts.index = np.asarray(obs["CellName"])
    
    adata = anndata.AnnData(counts, dtype=np.float32, obs = obs, var = var )
    
    adata.var["Count"] = adata.X.sum(axis=0)
    genes = pd.read_excel(genemetafile).fillna("")
    genes.index = [gene.upper() if sp!="Mouse" else gene.upper()+"_M" for gene, sp in zip(genes["Gene"], genes["Species"])]
    adata.var = pd.merge(adata.var,genes,left_index=True,right_index=True,how="left")#.sort_values("Count",ascending=False)
    adata = adata[:,adata.var.sort_values("Count",ascending=False).index].copy()
    
    adata.obs["TotalGeneCount"] = counts.sum(axis=1)
    adata.obs["MouseGeneCount"] = counts[adata.var.loc[adata.var["Species"]=="Mouse","GeneR"]].sum(axis=1)
    adata.obs["HumanGeneCount"] = counts[adata.var.loc[adata.var["Species"]=="Human","GeneR"]].sum(axis=1)
    adata.obs["MouseGeneShare"] = counts[adata.var.loc[adata.var["Species"]=="Mouse","GeneR"]].sum(axis=1)/counts.sum(axis=1)
    adata.obs["HumanGeneShare"] = counts[adata.var.loc[adata.var["Species"]=="Human","GeneR"]].sum(axis=1)/counts.sum(axis=1)
    
    merged =  pd.merge(adata.obs,pd.read_table(resultsfolder+"/segmentation_cell_stats.csv", sep=",").rename(columns={"cell":"MaskIndex"}),
                            left_on="MaskIndex", right_on="MaskIndex")
    merged.index = adata.obs.index
    adata.obs = merged
    
    return adata

##############################
### Combine transcripts from multiple ROIs
##############################

def combine_baysor_transcripts(files, outfile, shift=5000, cellshift=50000):
    """ Combines files, adds shift to x and cellshift to cell, saves parameters in ..._combinekey.npz.
    """
    roikeys = [re.search(r'R(\d)_W(\d)A(\d)', filename).group(0) for filename in files]
    def load_and_addxc(file, x, c):
        counts = pd.read_table(file, sep=",")
        counts["x"] += x
        if "cell" in counts.columns:
            counts.loc[counts["cell"]!=0, "cell"] += c
        return counts
    counts = [load_and_addxc(file, i*shift, i*cellshift) for i, file in enumerate(files)]
    boundaries = np.arange(1, len(counts)+1)*shift
    cellboundaries = np.arange(1, len(counts)+1)*cellshift
    combined = pd.concat(counts)
    
    combined.to_csv(outfile, index=False)
    np.savez_compressed(outfile.replace(".csv","_combinekey.npz"),
                        roikeys = roikeys, boundaries = boundaries, cellboundaries = cellboundaries)

##############################
### Combined ROIs
##############################

def load_multiple_ROIs_transcripts(resultfolder, genemetafile, do_correction=False):
    """ Load transcripts for run that contains multiple ROIs.
    """
    meta = pd.read_table(genemetafile, sep=",", index_col=0)
    transcripts_wnoise = pd.read_table(resultfolder+"/segmentation.csv", sep=",")
    if do_correction:
        transcripts_wnoise.loc[transcripts_wnoise["gene"]=='H2-K1', "gene"] = 'H2-K1_M'
        transcripts_wnoise.loc[transcripts_wnoise["gene"]=='MARCH4', "gene"] = 'MARCHF4'
        transcripts_wnoise.loc[transcripts_wnoise["gene"]=='PDGFRA', "gene"] = 'PDGFRA_M'
    transcripts_wnoise["celltype"] = np.asarray((meta["Species"] + " - " + \
                                          meta["Celltype"]).loc[transcripts_wnoise["gene"]])
    transcripts_wnoise["celltypegene"] = transcripts_wnoise["celltype"] + " - " + transcripts_wnoise["gene"]
    transcripts = transcripts_wnoise[~transcripts_wnoise["is_noise"]].copy().reset_index(drop=True)
    return transcripts, transcripts_wnoise

def cluster_crosstab(trans, norm = True, normgenes = True, wnoise = True, comparekey = "celltype", wtotal=True):
    """ Crosstab of Baysor transcripts with assigned cluster.
    """
    cross = pd.crosstab(trans[comparekey], trans["cluster"])
    total = cross.sum(axis=1)
    if not norm: return cross
    if normgenes:
        cross = np.round(cross/np.asarray(cross.sum(axis=1))[:,None]*100,0).astype(int)
        cross = cross.astype(str)
        cross[cross=="0"] = ""
        if wtotal: cross["total"] = total
        return cross
    else:
        cross = np.round(cross/np.asarray(cross.sum(axis=0))[None]*100,0).astype(int)
        cross = cross.astype(str)
        cross[cross=="0"] = ""
        if wtotal: cross["total"] = total
        return cross

def cluster_combine(transcripts, transcripts_wnoise, clusterlist=[]):
        """ Combine clusters.
            Order of the combinations in clusterlist is irrelevant, will
            always reduce all equivalent clusters to that with the lowest index.
        """
        Ninit = np.unique(transcripts["cluster"]).shape[0]
        clusterlistsorted = [sorted(l, reverse=True) for l in sorted(clusterlist, key=max, reverse=True)]
        replacedict = dict(zip(np.arange(transcripts["cluster"].max()+1),np.arange(transcripts["cluster"].max()+1)))
        for repl_ in clusterlistsorted:
            repl = [replacedict[k] for k in repl_]
            for key in replacedict:
                if replacedict[key] in repl:
                    replacedict[key] = min(repl)
        for key in replacedict:
            transcripts.loc[transcripts["cluster"]==key, "cluster"] = replacedict[key]
            transcripts_wnoise.loc[transcripts_wnoise["cluster"]==key, "cluster"] = replacedict[key]

def save_clusterids(resultfolder, cluster_combine_list, clusternamedict):
    """ Save information on clusters for Baysor run.
    """
    np.savez_compressed(resultfolder+"/cluster_ids.npz",
                        cluster_combine_list = cluster_combine_list,
                        clusternamedict = clusternamedict)

def split_baysor_ROIs(resultfolder, keyfile):
    """ Split Baysor results into ROIs.
        Creates folder structure resultsfolder/rois/...
    """
    file = np.load(keyfile)
    roikeys = file["roikeys"]
    boundaries = file["boundaries"]
    transcripts = pd.read_table(resultfolder+"/segmentation.csv", sep=",")
    cells = pd.read_table(resultfolder+"/segmentation_cell_stats.csv", sep=",")
    counts = pd.read_table(resultfolder+"/segmentation_counts.tsv", sep="\t", index_col=0)
    counts.columns = counts.columns.astype(int)

    def split_df(dffull_, boundaries):
        dffull = dffull_.copy()
        dffull["roi"] = np.searchsorted(boundaries, dffull["x"])
        roi_dfs = []
        for i in range(len(boundaries)):
            df = dffull.loc[dffull["roi"]==i].loc[:,dffull.columns!="roi"].copy()
            if i>0: df["x"] -= boundaries[i-1]
            roi_dfs.append(df)
        return roi_dfs

    roi_transcripts = split_df(transcripts, boundaries)
    roi_cells = split_df(cells, boundaries)

    for i in range(len(roikeys)):
        path = resultfolder+"/rois/"+roikeys[i]
        if not os.path.exists(path):
            os.makedirs(path)
        roi_transcripts[i].to_csv(path+"/segmentation.csv", sep=",", index=False)
        roi_cells[i].to_csv(path+"/segmentation_cell_stats.csv", sep=",", index=False)
        counts[roi_cells[i]["cell"]].to_csv(path+"/segmentation_counts.csv", sep="\t")



