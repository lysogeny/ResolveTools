import numpy as np
import pandas as pd
import anndata
import re
import os
from scipy.spatial import distance_matrix

from ..resolve.resolveimage import ResolveImage, read_genemeta_file
from ..utils.utils import printwtime
from ..segmentation.counts import read_loom
from ..utils.parameters import CONFOCAL_VOXEL_SIZE
from ..image.utils import read_single_modality_confocal

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

def assign_counts_from_Baysor(resultsfolder, genemetafile, roikey, do_for="cell", clusteridfile=""):
    """ Assign counts to cells, using Baysor output.
    """
    if not do_for in ["cell", "cluster"]: raise ValueError("Not available")
    
    segmentation = pd.read_table(resultsfolder+"/segmentation.csv", sep=",")
    segmentation = segmentation[~segmentation["is_noise"]]
    
    var = pd.DataFrame(np.unique(segmentation["gene"]))
    var.columns = ["GeneR"]
    var.index = np.asarray(var["GeneR"])
    obs = pd.DataFrame(np.unique(segmentation[do_for]), columns=["MaskIndex"])
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
    genes = read_genemeta_file(genemetafile)
    adata.var = pd.merge(adata.var,genes,left_index=True,right_index=True,how="left")#.sort_values("Count",ascending=False)
    adata = adata[:,adata.var.sort_values("Count",ascending=False).index].copy()
    
    adata.obs["TotalGeneCount"] = counts.sum(axis=1)
    adata.obs["MouseGeneCount"] = counts[adata.var.loc[adata.var["Species"]=="Mouse","GeneR"]].sum(axis=1)
    adata.obs["HumanGeneCount"] = counts[adata.var.loc[adata.var["Species"]=="Human","GeneR"]].sum(axis=1)
    adata.obs["MouseGeneShare"] = counts[adata.var.loc[adata.var["Species"]=="Mouse","GeneR"]].sum(axis=1)/counts.sum(axis=1)
    adata.obs["HumanGeneShare"] = counts[adata.var.loc[adata.var["Species"]=="Human","GeneR"]].sum(axis=1)/counts.sum(axis=1)
    
    merged =  pd.merge(adata.obs,pd.read_table(resultsfolder+"/segmentation_cell_stats.csv", sep=",").rename(columns={"cell":"MaskIndex"}),
                            left_on="MaskIndex", right_on="MaskIndex")
    merged.index = np.asarray(merged["CellName"])
    if not (len(merged.index)>0.98*len(obs.index) or len(merged.index)<1.02*len(obs.index)):
        printwtime("Discreptancy between cell count in _cell_stats.csv and cells in segmentation.csv!")
        printwtime("  Merged has {}, original has {} cells.".format(len(merged.index), len(obs.index)))
    adata = adata[merged.index].copy() # In case there are any errors in comparison of 
    adata.obs = merged
    
    adata.obs["BaysorCluster"] = adata.obs["cluster"]
    
    if clusteridfile:
        cluster_combine_list = np.load(clusteridfile, allow_pickle=True)["cluster_combine_list"]
        clusternamedict = np.load(clusteridfile, allow_pickle=True)["clusternamedict"].item()
        
        clusterlistsorted = [sorted(l, reverse=True) for l in sorted(cluster_combine_list, key=max, reverse=True)]
        clusterNmax = max(max([max(l) for l in clusterlistsorted]), adata.obs["cluster"].max())
        replacedict = dict(zip(np.arange(clusterNmax+1),np.arange(clusterNmax+1)))
        for repl_ in clusterlistsorted:
            repl = [replacedict[k] for k in repl_]
            for key in replacedict:
                if replacedict[key] in repl:
                    replacedict[key] = min(repl)
        for key in replacedict:
            adata.obs.loc[adata.obs["BaysorCluster"]==key, "BaysorCluster"] = replacedict[key]
        
        if len(clusternamedict)>0:
            adata.obs["BaysorClusterCelltype"] = adata.obs["BaysorCluster"].apply(lambda x: clusternamedict[x])
    
    return adata

##############################
### Combine transcripts from multiple ROIs
##############################

def combine_baysor_transcripts(files, outfile, shift=10000, cellshift=100000, dropgenes=[]):
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
    
    if len(dropgenes)>0:
        combined = combined[combined["gene"].apply(lambda x: x not in dropgenes)]
    
    combined.to_csv(outfile, index=False)
    np.savez_compressed(outfile.replace(".csv","_combinekey.npz"),
                        roikeys = roikeys, boundaries = boundaries, cellboundaries = cellboundaries)

##############################
### Combined ROIs
##############################

def load_multiple_ROIs_transcripts(resultfolder, genemetafile, do_correction=True):
    """ Load transcripts for run that contains multiple ROIs.
    """
    meta = read_genemeta_file(genemetafile)
    transcripts_wnoise = pd.read_table(resultfolder+"/segmentation.csv", sep=",")
    if do_correction:
        transcripts_wnoise.loc[transcripts_wnoise["gene"]=='H2-K1', "gene"] = 'H2-K1_M'
        transcripts_wnoise.loc[transcripts_wnoise["gene"]=='MARCH4', "gene"] = 'MARCHF4'
        transcripts_wnoise.loc[transcripts_wnoise["gene"]=='PDGFRA', "gene"] = 'PDGFRA_M'
    transcripts_wnoise["celltype"] = np.asarray((meta["Species"] + " - " + meta["Celltype"]).loc[transcripts_wnoise["gene"]])
    transcripts_wnoise["celltypegene"] = transcripts_wnoise["celltype"] + " - " + transcripts_wnoise["gene"]
    transcripts = transcripts_wnoise[~transcripts_wnoise["is_noise"]].copy().reset_index(drop=True)
    return transcripts, transcripts_wnoise

def cluster_crosstab(trans, norm = True, normgenes = True, wnoise = True, comparekey = "celltype", wtotal=True, roundn=0):
    """ Crosstab of Baysor transcripts with assigned cluster.
    """
    cross = pd.crosstab(trans[comparekey], trans["cluster"])
    total = cross.sum(axis=1)
    if not norm: return cross
    if normgenes:
        cross = np.round(cross/np.asarray(cross.sum(axis=1))[:,None]*100,roundn)
        if roundn==0: cross = cross.astype(int)
        cross = cross.astype(str)
        cross[cross=="0"] = ""
        if wtotal: cross["total"] = total
        return cross
    else:
        cross = np.round(cross/np.asarray(cross.sum(axis=0))[None]*100,roundn).astype(int)
        if roundn==0: cross = cross.astype(int)
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

def save_clusterids(resultfolder, cluster_combine_list, clusternamedict, cross, humannamecolordict, mousenamecolordict):
    """ Save information on clusters for Baysor run.
    """
    np.savez_compressed(resultfolder+"/cluster_ids.npz",
                        cluster_combine_list = cluster_combine_list,
                        clusternamedict = clusternamedict,
                        cross = cross,
                        humannamecolordict = humannamecolordict,
                        mousenamecolordict = mousenamecolordict)

def find_cluster_correspondence(target, source, ignore_indices=[], cutoff = 3):
    """ Find corresponding Baysor clusters, using cluster_crosstab(..., wtotal=False) output.
    """
    target_f = np.delete(np.asarray(target.replace("","0").astype(int).T), ignore_indices, axis=1)
    source_f = np.delete(np.asarray(source.replace("","0").astype(int).T), ignore_indices, axis=1)
    dist = distance_matrix(target_f, source_f)
    repl = dist.argmin(axis=1)
    if len(repl) != len(np.unique(repl)):
        raise ValueError("Found no clear correspondence of cluster labels! Not every cluster has a clear correspondence.")
    test = dist[np.arange(len(repl)),repl]
    if test.max()>cutoff:
        raise ValueError("Found no clear correspondence of cluster labels! Cluster {} has error {}.".format(test.argmax(), test.max()))
    return np.argsort(repl)+1

def split_baysor_ROIs(resultfolder, keyfile, idfile="", genemetafile="", do_correction=True, ignore_indices=[], safetyshift=5, correspondwnoise=True):
    """ Split Baysor results into ROIs.
        Creates folder structure resultsfolder/rois/...
        Corresponds cluster labels to that from idfile if provided.
    """
    file = np.load(keyfile)
    roikeys = file["roikeys"]
    boundaries = file["boundaries"]
    cellboundaries = file["cellboundaries"]
    transcripts = pd.read_table(resultfolder+"/segmentation.csv", sep=",")
    if do_correction:
        transcripts.loc[transcripts["gene"]=='H2-K1', "gene"] = 'H2-K1_M'
        transcripts.loc[transcripts["gene"]=='MARCH4', "gene"] = 'MARCHF4'
        transcripts.loc[transcripts["gene"]=='PDGFRA', "gene"] = 'PDGFRA_M'
    
    cells = pd.read_table(resultfolder+"/segmentation_cell_stats.csv", sep=",")
    counts = pd.read_table(resultfolder+"/segmentation_counts.tsv", sep="\t", index_col=0)
    counts.columns = counts.columns.astype(int)
    
    if idfile and genemetafile:
        meta = read_genemeta_file(genemetafile)
        transcripts["celltype"] = np.asarray((meta["Species"] + " - " + meta["Celltype"]).loc[transcripts["gene"]])
        target = pd.DataFrame(np.load(idfile, allow_pickle=True)["cross"])
        source = cluster_crosstab(transcripts if correspondwnoise else transcripts[~transcripts["is_noise"]], wtotal=False)
        repl = find_cluster_correspondence(target, source, ignore_indices=ignore_indices)
        transcripts = transcripts.loc[:,transcripts.columns!="celltype"].copy()
        
        transcripts["cluster"] = repl[transcripts["cluster"]-1]
        cells["cluster"] = repl[cells["cluster"]-1]

    def split_df(dffull_, boundaries, cellboundaries):
        dffull = dffull_.copy()
        dffull["roi"] = np.searchsorted(boundaries, np.round(dffull["x"],1), side="right")
        roi_dfs = []
        for i in range(len(boundaries)):
            df = dffull.loc[dffull["roi"]==i].loc[:,dffull.columns!="roi"].copy()
            if i<len(boundaries)-1:
                df = df.loc[df["x"]<boundaries[i]-safetyshift]
            if i>0:
                df["x"] -= boundaries[i-1]
                if "prior_segmentation" in df.columns:
                    df.loc[df["prior_segmentation"]!=0, "prior_segmentation"] -= cellboundaries[i-1]
                
            roi_dfs.append(df)
        return roi_dfs

    roi_transcripts = split_df(transcripts, boundaries, cellboundaries)
    roi_cells = split_df(cells, boundaries, cellboundaries)

    for i in range(len(roikeys)):
        path = resultfolder+"/rois/"+roikeys[i]
        if not os.path.exists(path):
            os.makedirs(path)
        roi_transcripts[i].to_csv(path+"/segmentation.csv", sep=",", index=False)
        roi_cells[i].to_csv(path+"/segmentation_cell_stats.csv", sep=",", index=False)
        counts[roi_cells[i]["cell"]].to_csv(path+"/segmentation_counts.csv", sep="\t")

def combine_adatas(resultfolder, genemetafile, loomfile, outfile=""):
    """ Combine .loom file for all ROIs in resultfolder.
    """
    path = resultfolder+"/rois/"
    rois = os.listdir(path)
    genemeta = read_genemeta_file(genemetafile)
    def read_adata(file):
        adata = read_loom(file)
        adata.obs.index = list(adata.obs["CellName"])
        return adata
    adatas = [read_adata(path+roi+"/"+loomfile) for roi in rois]
    def concat_adatas(adatas, genemeta):
        adata = anndata.concat(adatas, join="outer", merge="first")
        adata.var = genemeta.loc[adata.var.index]
        return adata
    adata = concat_adatas(adatas, genemeta)
    if outfile: adata.write_loom(resultfolder+"/"+outfile)
    return adata

def split_segmentation_counts_ROI(segmentation, keyfile):
    """ Takes combined segmentation.csv and keyfile, 
        adds ROI and removes shift of prior_segmentation and x.
    """
    file = np.load(keyfile)
    roikeys = file["roikeys"]
    boundaries = file["boundaries"]
    cellboundaries = file["cellboundaries"]
    segmentation["ROI"] = np.searchsorted(boundaries, np.round(segmentation["x"],1), side="right")
    for i in range(len(boundaries)):
        if i>0:
            segmentation.loc[segmentation["ROI"]==i, "x"] -= boundaries[i-1]
            if "prior_segmentation" in segmentation.columns:
                segmentation.loc[np.logical_and(segmentation["ROI"]==i, segmentation["prior_segmentation"]!=0),
                                 "prior_segmentation"] -= cellboundaries[i-1]
    segmentation["ROI"] = segmentation["ROI"].apply(lambda x: roikeys[x])

def split_transcripts_assigned(resultfolder, keyfile, genemetafile):
    """ Takes segmentation.csv, adds ROI, assigned segcell etc.
    """
    segmentation_wnoise = pd.read_table(resultfolder+"/segmentation.csv", sep=",")

    adatabay = read_loom(resultfolder+"/"+os.path.basename(resultfolder)+"_baysor_cells.loom")
    #adatabay.obs.index = adatabay.obs["CellName"]
    obsbay = adatabay.obs
    obsbay.index = obsbay["MaskIndex"]

    adataseg = read_loom(resultfolder+"/"+os.path.basename(resultfolder)+"_segmentation_cells.loom")
    #adataseg.obs.index = adataseg.obs["CellName"]
    obsseg = adataseg.obs
    
    split_segmentation_counts_ROI(segmentation_wnoise, keyfile)
    
    obsbay["assigned_to_str"] = ""
    mask = obsbay.index[obsbay["assigned_to"]!=0]
    obsbay.loc[mask, "assigned_to_str"] = obsbay.loc[mask, "ROI"]+"_"+obsbay.loc[mask, "assigned_to"].astype(str)

    segmentation_wnoise["assigned_to_str"] = ""
    assignedindex = segmentation_wnoise.index[segmentation_wnoise["cell"]!=0]
    segmentation_wnoise.loc[assignedindex, "assigned_to_str"] = \
            list(obsbay.loc[segmentation_wnoise.loc[assignedindex, "cell"], "assigned_to_str"])

    hassegindex = segmentation_wnoise.index[segmentation_wnoise["assigned_to_str"]!=""]

    segmentation_wnoise["to_x"] = 0
    segmentation_wnoise.loc[hassegindex, "to_x"] = list(obsseg.loc[segmentation_wnoise.loc[hassegindex, "assigned_to_str"],"x"])
    segmentation_wnoise["to_y"] = 0
    segmentation_wnoise.loc[hassegindex, "to_y"] = list(obsseg.loc[segmentation_wnoise.loc[hassegindex, "assigned_to_str"],"y"])
    
    genemeta = read_genemeta_file(genemetafile)
    segmentation_wnoise["celltype"] = np.asarray((genemeta["Species"] + " - " + \
                                              genemeta["Celltype"]).loc[segmentation_wnoise["gene"]])
    
    segmentation_wnoise.to_csv(resultfolder+"/segmentation_assigned.csv", sep=",", index=False)

##############################
### Add Stain to ROI
##############################

def add_stain_to_ROI(resultfolder, imagefolder, segmentationfolder, roi, extend=[1,10,10]):
    """ Add image stains to single ROI cells.
    """
    printwtime("Adding stains to ROI "+roi)
    
    segmentation_wnoise = pd.read_table(resultfolder+"/rois/"+roi+"/segmentation.csv", sep=",")
    segmentation = segmentation_wnoise[~segmentation_wnoise["is_noise"]].copy()
    segmentation = segmentation[segmentation["cell"]!=0].copy()
    segmentation["cell"] = roi+"_"+segmentation["cell"].astype(str)
    segmentation[["z","y","x"]] = (segmentation[["z","y","x"]]/np.asarray(CONFOCAL_VOXEL_SIZE[:3])).astype(int)

    adatabay = read_loom(resultfolder+"/rois/"+roi+"/baysor_cells_post.loom")
    adatabay.obs.index = adatabay.obs["CellName"]
    obsbay = adatabay.obs
    adataseg = read_loom(resultfolder+"/rois/"+roi+"/segmentation_cells.loom")
    adataseg.obs.index = adataseg.obs["CellName"]
    obsseg = adataseg.obs

    def sample_extended(segmentation, image, key, extend=[1,10,10]):
        zmesh, ymesh, xmesh = np.meshgrid(  np.arange(0-extend[0],1+extend[0]),
                                            np.arange(0-extend[1],1+extend[1]),
                                            np.arange(0-extend[2],1+extend[2]))
        zcoord = np.clip(zmesh[None] + np.asarray(segmentation["z"])[:,None,None,None], 0, image.shape[0]-1)
        ycoord = np.clip(ymesh[None] + np.asarray(segmentation["y"])[:,None,None,None], 0, image.shape[1]-1)
        xcoord = np.clip(xmesh[None] + np.asarray(segmentation["x"])[:,None,None,None], 0, image.shape[2]-1)

        segmentation[key] = image[zcoord, ycoord, xcoord].mean(axis=-1).mean(axis=-1).mean(axis=-1)

    for mod in ["DAPI","WGA","mCherry","EGFP","IB4","HCM"]:
        printwtime("  Loading "+mod)
        impath = imagefolder+"/Confocal_"+roi+"_"+mod+".tif"
        if os.path.exists(impath):
            image = read_single_modality_confocal(impath)
            sample_extended(segmentation, image, "TCFmean_"+mod, extend=[0,0,0])
            sample_extended(segmentation, image, "TCFmean_"+mod+"_ext", extend=extend)
            del image
    
    def seg_to_obsbay(key):
        obsbay[key] = segmentation.groupby("cell").apply(lambda x: x[key].mean())
    def obsbay_to_obsseg(key):
        means = obsbay[obsbay["assigned_to"]!=0].groupby("assigned_to").apply(
                           lambda x: (x["n_transcripts"]*x[key]).sum()/x["n_transcripts"].sum())
        means.index = roi+"_"+means.index.astype(str)
        obsseg[key] = means
    
    for key in segmentation.columns:
        if "TCFmean_" in key:
            #print(key)
            seg_to_obsbay(key)
            obsbay_to_obsseg(key)
    
    meta = pd.read_table(segmentationfolder+"/Confocal_"+roi+"_DAPI_mesmer_nuclei_post_meta.csv", sep=",", index_col=0)
    for key in meta.columns:
        if "CFmean_" in key:
            obsseg[key] = meta[key]
    
    adatabay.write_loom(resultfolder+"/rois/"+roi+"/baysor_cells_post.loom")
    adataseg.write_loom(resultfolder+"/rois/"+roi+"/segmentation_cells.loom")



