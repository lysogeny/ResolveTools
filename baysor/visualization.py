import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..segmentation.visualize import get_rgb_distinct

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
    
    cols = get_rgb_distinct(obs["cluster"].max())
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
