import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from ..segmentation.visualize import get_rgb_distinct
from ..segmentation.counts import read_loom
from ..image.utils import read_single_modality_confocal
from ..utils.parameters import CONFOCAL_VOXEL_SIZE
from ..utils.utils import printwtime
from .utils import split_transcripts_assigned

##############################
### Simple Cell Plot
##############################

def plot_celltypedist(cellloomfile, segmetafile, idfile, outfile="", title=""):
    """ Plot cells, colored by celltype color from idfile.
    """
    adata = read_loom(cellloomfile)
    seg = pd.read_table(segmetafile, sep=",", index_col=0)
    humannamecolordict = np.load(idfile, allow_pickle=True)["humannamecolordict"].item()
    mousenamecolordict = np.load(idfile, allow_pickle=True)["mousenamecolordict"].item()
    
    xmax, ymax = adata.obs["x"].max(), adata.obs["y"].max()
    fig, ax = plt.subplots(2,1,figsize=(int(6*xmax/500), int(6*2*ymax/500)))
    
    sns.scatterplot(x=seg["x"], y=seg["y"], color="gray", ax=ax[0], legend=False, s=10, alpha=0.5)
    sns.scatterplot(x=seg["x"], y=seg["y"], color="gray", ax=ax[1], legend=False, s=10, alpha=0.5)
    
    for key in humannamecolordict:
        mask = adata.obs["BaysorClusterCelltype"]==key
        sns.scatterplot(x=adata.obs.loc[mask, "x"], y=adata.obs.loc[mask, "y"],
                        color=humannamecolordict[key], label=key, ax=ax[0], legend=False, s=10, alpha=1)
    ax[0].legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1), facecolor='white', framealpha=1)
    
    for key in mousenamecolordict:
        mask = adata.obs["BaysorClusterCelltype"]==key
        sns.scatterplot(x=adata.obs.loc[mask, "x"], y=adata.obs.loc[mask, "y"],
                        color=mousenamecolordict[key], label=key, ax=ax[1], legend=False, s=10, alpha=1)
    ax[1].legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1), facecolor='white', framealpha=1)
    
    if title: plt.suptitle(title, size=30)
    if outfile:
        plt.savefig(outfile, dpi=300)
        plt.close()

##############################
### Final Assignment Plot
##############################

def plot_final_assignment(roidata, background="", file="", dpi=900):
    """ Plot final assignment of transcripts to clusters, and clusters to cells.
        
        SHOWS TRANSCRIPTS AS CLUSTER IDENTITY, LOOK AT D AGAIN!!!!!
    """
    obs = roidata.obsbay.copy()
    transcripts = roidata.transcripts
    transcripts_noise = roidata.transcripts_wnoise[roidata.transcripts_wnoise["is_noise"]]
    
    xt = np.asarray(roidata.transcripts["x"])
    yt = np.asarray(roidata.transcripts["y"])
    xc = np.asarray(obs.loc[roidata.transcripts["cell"],"x"])
    yc = np.asarray(obs.loc[roidata.transcripts["cell"],"y"])
    
    assignment = roidata._cgetbay("segcell")[roidata._cgetbay("segcell")!=0]
    
    obs_seg = roidata.obsseg.copy()
    
    xt_assign = np.asarray(obs.loc[list(assignment.index),"x"])
    yt_assign = np.asarray(obs.loc[list(assignment.index),"y"])
    xc_assign = np.asarray(obs_seg.loc[list(assignment),"x"])
    yc_assign = np.asarray(obs_seg.loc[list(assignment),"y"])
    
    cols = get_rgb_distinct(np.max([obs["cluster"].max(), transcripts["cluster"].max(), transcripts_noise["cluster"].max()]))
    xmax, ymax = transcripts["x"].max(), transcripts["y"].max()
    fig, ax = plt.subplots(1,1,figsize=(xmax/50,ymax/50))
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, ymax])
    
    if background:
        image = read_single_modality_confocal(background)[0]
        size = image.shape*np.asarray(CONFOCAL_VOXEL_SIZE[1:3])
        plt.imshow(image, origin="lower", cmap="gray", extent=[0, size[1], 0, size[0]], alpha=0.5)
        del image
    
    sns.scatterplot(x=obs_seg["x"], y=obs_seg["y"], color="black", s=3, ax=ax)
    sns.scatterplot(x=obs["x"], y=obs["y"], color=cols[list(obs["cluster"]-1)], s=3, ax=ax)
    sns.scatterplot(x=transcripts["x"], y=transcripts["y"],
                    color=cols[list(transcripts["cluster"]-1)], s=1, ax=ax, alpha=0.7)
    sns.scatterplot(x=transcripts_noise["x"], y=transcripts_noise["y"],
                    color=cols[list(transcripts_noise["cluster"]-1)], s=0.5, ax=ax, alpha=0.7)
    
    xx = np.vstack([xt,xc])
    yy = np.vstack([yt,yc])
    plt.plot(xx,yy, '-', color="dimgray", linewidth=0.15)
    
    xx = np.vstack([xt_assign,xc_assign])
    yy = np.vstack([yt_assign,yc_assign])
    plt.plot(xx, yy, '-', color="black", linewidth=0.2)
    
    sns.scatterplot(x=obs_seg["x"], y=obs_seg["y"], color="black", s=0.5, ax=ax)
    
    if file: plt.savefig(file, dpi = dpi, bbox_inches ="tight")

def plot_final_assignment_post_singleROI(segmentation_wnoise, obssegPT, obsseg, genecolors, celltypecolors,
                                         background="", file="", dpi=900):
    """ Plot single ROI.
    """
    segmentation_wnoise["color"] = segmentation_wnoise["celltype"].apply(lambda x: genecolors[x])
    obssegPT["color"] = obssegPT["BaysorClusterCelltype"].apply(lambda x: celltypecolors[x])

    obsfull = obsseg
    obs = obssegPT
    transcripts = segmentation_wnoise[segmentation_wnoise["assigned_to_str"]==segmentation_wnoise["assigned_to_str"]]
    transcripts_noise = segmentation_wnoise[segmentation_wnoise["assigned_to_str"]!=segmentation_wnoise["assigned_to_str"]]

    obsfull["not_empty"] = False
    obsfull.loc[obs.index, "not_empty"] = True
    obsempty = obsfull[~obsfull["not_empty"]]

    xmax, ymax = transcripts["x"].max(), transcripts["y"].max()
    fig, ax = plt.subplots(1,1,figsize=(xmax/50,ymax/50))
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, ymax])
    
    if background:
        image = read_single_modality_confocal(background)[0]
        size = image.shape*np.asarray(CONFOCAL_VOXEL_SIZE[1:3])
        plt.imshow(image, origin="lower", cmap="gray", extent=[0, size[1], 0, size[0]], alpha=0.5)
        del image

    sns.scatterplot(x=transcripts["x"], y=transcripts["y"], color=transcripts["color"], s=1, ax=ax, alpha=0.7)
    sns.scatterplot(x=transcripts_noise["x"], y=transcripts_noise["y"], color=transcripts_noise["color"], s=0.5, ax=ax, alpha=0.7)

    xt = np.asarray(transcripts["x"])
    yt = np.asarray(transcripts["y"])
    xc = np.asarray(transcripts["to_x"])
    yc = np.asarray(transcripts["to_y"])

    xx = np.vstack([xt,xc])
    yy = np.vstack([yt,yc])
    plt.plot(xx, yy, '-', color="dimgray", linewidth=0.15)

    sns.scatterplot(x=obsempty["x"], y=obsempty["y"], color="gray", s=2, ax=ax, zorder=2.5)
    sns.scatterplot(x=obs["x"], y=obs["y"], color=obs["color"], s=2, ax=ax, zorder=2.5)

    if file: plt.savefig(file, dpi = dpi, bbox_inches ="tight")

human_genecolors = {
        'Human - Activation':"red",
        'Human - Differentiation':"purple",
        'Human - Human':"blue",
        'Human - Quiescence':"green",
        'Human - Quiescence/Activation':"orange",
        'Human/Mouse - Activation':"red",
        'Mouse - Astrocyte':"black",
        'Mouse - Endothelial':"black",
        'Mouse - Macrophages':"black",
        'Mouse - Macrophages/Microglia':"black",
        'Mouse - Microglia':"black",
        'Mouse - Mouse':"black",
        'Mouse - Neuron':"black",
        'Mouse - OD':"black",
        'Mouse - OPC/OD':"black",
        'Mouse - OPC':"black"
}
mouse_genecolors = {
        'Human - Activation':"black",
        'Human - Differentiation':"black",
        'Human - Human':"black",
        'Human - Quiescence':"black",
        'Human - Quiescence/Activation':"black",
        'Human/Mouse - Activation':"black",
        'Mouse - Astrocyte':"blue",
        'Mouse - Endothelial':"red",
        'Mouse - Macrophages':"yellow",
        'Mouse - Macrophages/Microglia':"yellow",
        'Mouse - Microglia':"yellow",
        'Mouse - Mouse':"orange",
        'Mouse - Neuron':"green",
        'Mouse - OD':"purple",
        'Mouse - OPC/OD':"purple",
        'Mouse - OPC':"purple"
}
human_celltypecolors = {
        'Astrocyte':"black",
        'Endothelial':"black",
        'Human A':"red",
        'Human D':"purple",
        'Human Q':"green",
        'Human Q/A':"orange",
        'Human':"blue",
        'Macrophages':"black",
        'Microglia':"black",
        'Neuron':"black",
        'OD':"black",
        'unknown':"gray"
}
mouse_celltypecolors = {
        'Astrocyte':"blue",
        'Endothelial':"red",
        'Human A':"black",
        'Human D':"black",
        'Human Q':"black",
        'Human Q/A':"black",
        'Human':"black",
        'Macrophages':"yellow",
        'Microglia':"yellow",
        'Neuron':"green",
        'OD':"purple",
        'unknown':"gray"
}

def plot_final_assignment_post(resultfolder, keyfile, genemetafile, backgroundtemplate="", humanbackgroundtemplate="", dpi=900, onlyhumanbackground=False, excluderois=[]):
    printwtime("Split segmentation.csv")
    split_transcripts_assigned(resultfolder, keyfile, genemetafile)
    
    segmentation_wnoise = pd.read_table(resultfolder+"/segmentation_assigned.csv", sep=",")
    adataseg = read_loom(resultfolder+"/"+os.path.basename(resultfolder)+"_segmentation_cells.loom")
    obsseg = adataseg.obs
    adatasegPT = read_loom(resultfolder+"/"+os.path.basename(resultfolder)+"_segmentation_cells_QC.loom")
    obssegPT = adatasegPT.obs
    
    path = resultfolder+"/final_assignment_plots/"
    if not os.path.exists(path): os.makedirs(path)
    
    printwtime("Create final assignment plots")
    rois = list(filter(lambda x: x not in excluderois, os.listdir(resultfolder+"/rois/")))
    for roi in rois:
        printwtime("  Plotting final assignment for ROI "+roi)
        if not onlyhumanbackground:
            plot_final_assignment_post_singleROI(segmentation_wnoise[segmentation_wnoise["ROI"]==roi].copy(),
                                                 obssegPT[obssegPT["ROI"]==roi].copy(),
                                                 obsseg[obsseg["ROI"]==roi].copy(),
                                                 human_genecolors, human_celltypecolors,
                                                 background="", file=path+roi+"_final_assignment_human.jpg", dpi=dpi)
            plt.close()
            if backgroundtemplate:
                plot_final_assignment_post_singleROI(
                                                     segmentation_wnoise[segmentation_wnoise["ROI"]==roi].copy(),
                                                     obssegPT[obssegPT["ROI"]==roi].copy(),
                                                     obsseg[obsseg["ROI"]==roi].copy(),
                                                     human_genecolors, human_celltypecolors,
                                                     background=backgroundtemplate.format(roi), file=path+roi+"_final_assignment_human_wbackground.jpg", dpi=dpi)
                plt.close()
        
        if humanbackgroundtemplate:
            mode = "mCherry" if os.path.isfile(humanbackgroundtemplate.format(roi,"mCherry")) else "HCM"
            plot_final_assignment_post_singleROI(
                                                 segmentation_wnoise[segmentation_wnoise["ROI"]==roi].copy(),
                                                 obssegPT[obssegPT["ROI"]==roi].copy(),
                                                 obsseg[obsseg["ROI"]==roi].copy(),
                                                 human_genecolors, human_celltypecolors,
                                                 background=humanbackgroundtemplate.format(roi, mode), file=path+roi+"_final_assignment_human_whumanbackground.jpg", dpi=dpi)
            plt.close()
        
        if not onlyhumanbackground:
            plot_final_assignment_post_singleROI(segmentation_wnoise[segmentation_wnoise["ROI"]==roi].copy(),
                                                 obssegPT[obssegPT["ROI"]==roi].copy(),
                                                 obsseg[obsseg["ROI"]==roi].copy(),
                                                 mouse_genecolors, mouse_celltypecolors,
                                                 background="", file=path+roi+"_final_assignment_mouse.jpg", dpi=dpi)
            plt.close()
            if backgroundtemplate:
                plot_final_assignment_post_singleROI(
                                                     segmentation_wnoise[segmentation_wnoise["ROI"]==roi].copy(),
                                                     obssegPT[obssegPT["ROI"]==roi].copy(),
                                                     obsseg[obsseg["ROI"]==roi].copy(),
                                                     mouse_genecolors, mouse_celltypecolors,
                                                     background=backgroundtemplate.format(roi), file=path+roi+"_final_assignment_mouse_wbackground.jpg", dpi=dpi)
                plt.close()



