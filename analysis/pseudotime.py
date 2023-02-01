import numpy as np
import anndata
import pandas as pd
import glmpca.glmpca as glmpca
import matplotlib.pyplot as plt
import seaborn as sns
from skmisc.loess import loess

from ..utils.utils import printwtime

##############################
### Utils
##############################

def get_human_glmpca_data(adata, N=3):
    """ Reduce full adata to human cells for pseudotime estimation.
    """
    adatapca = adata[np.logical_and(adata.obs["TotalGeneCount"]>N, adata.obs["IS_HUMAN"])]
    counts = np.asarray(adatapca.X.todense()).T[adatapca.var["Species"]!="Mouse"]

    df = adatapca.to_df().T
    df.index = adatapca.var["GeneClass"]
    df = df.groupby("GeneClass").apply(lambda x: x.sum(axis=0))
    df = df.T
    adatapcashort = anndata.AnnData(df, obs=adatapca.obs).copy()
    adatapcashort.var["Gene"] = list(adatapcashort.var.index)
    adatapcashort = adatapcashort[adatapcashort.X.sum(axis=1)>0].copy()
    countsshort = np.asarray(adatapcashort.X).T[adatapcashort.var["Gene"].apply(lambda x: "Human" in x)]
    adatapcashort.X = adatapcashort.X/adatapcashort.X.sum(axis=1)[:,None]
    return countsshort, adatapcashort

##############################
### PT
##############################

def get_pseudotime(countsshort, adatapcashort, N=2, verbose=False, separation=0.3, tryFull=False):
    """ Get PT using glmpca.
    """
    printwtime("    Run glm-pca")
    factorsshort = glmpca.glmpca(countsshort, N, fam="nb", verbose=verbose)
    ctypesordered = ['Human Q', 'Human Q/A', 'Human A', 'Human D']
    def get_single_PT(i):
        pt = np.argsort(np.argsort(factorsshort["factors"][:,i]))
        pt = pt/pt.max()-pt.min()
        def meanstd(x):
            return x.mean(), x.std()
        means = np.asarray([meanstd(pt[adatapcashort.obs["BaysorClusterCelltype"]==ct]) for ct in ctypesordered])
        return means, pt
    for i in ([0] if not tryFull else range(N)):
        means, pt = get_single_PT(i)
        diff = (means[1:,0]-means[:-1,0])/np.sqrt(means[1:,1]**2+means[:-1,1]**2)
        printwtime("    "+str(i)+" "+str(diff))
        if np.all(diff<-separation):
            return True, 1-pt, i
        if np.all(diff>separation):
            return True, pt, i
    return False, np.zeros(pt.shape), -1

def get_meanpseudotime(countsshort, adatapcashort, meanN=10, N=2, verbose=False, separation=0.3, tryFull=False):
    """ Get PT multiple times, final PT from mean.
    """
    finalptlist = list()
    finalptindex = list()
    c = 0
    while len(finalptlist)<meanN:
        printwtime("Run "+str(c))
        c += 1
        success, pt, i = get_pseudotime(countsshort, adatapcashort, N=N, verbose=verbose, separation=separation, tryFull=tryFull)
        if success:
            finalptlist.append(pt)
            finalptindex.append(i)
            printwtime("  Found PT in component "+str(i))
        else:
            printwtime("  Found no PT")
    finalptlist = np.asarray(finalptlist)
    finalpt = finalptlist.mean(axis=0)
    finalpt = np.argsort(np.argsort(finalpt))
    finalpt = finalpt/finalpt.max()-finalpt.min()
    return finalpt, finalptlist, finalptindex

##############################
### PT Plot
##############################

def plot_PT_shares(pt, adatapcashort, hue="BaysorClusterCelltype"):
    """ Plot human gene changes over PT.
    """
    def plot_single(pt, groups, ax, name=""):
        yvals = np.asarray(sum([adatapcashort[:,s].X for s in groups]))[:,0]
        jitter = np.random.normal(0,0.005,len(pt))
        sns.scatterplot(x=pt, y=jitter+yvals, ax=ax, s=1,
                        hue=adatapcashort.obs[hue])
        smoother = loess(pt, yvals, span=0.4, degree=2)
        xout = np.arange(0,1.001,0.001)
        yout = smoother.predict(xout).values
        sns.lineplot(x=xout, y=yout, ax=ax, color="black")
        ax.set_title(groups[0] if not name else name)

    fig, ax = plt.subplots(3,2,figsize=(20,20))
    plot_single(pt, ["Human - Quiescence"], ax[0,0], name="Human Q")
    plot_single(pt, ["Human - Quiescence/Activation"], ax[0,1], name="Human Q/A")
    plot_single(pt, ["Human - Activation", "Human/Mouse - Activation"], ax[1,0], name="Human A")
    plot_single(pt, ["Human - Differentiation"], ax[1,1], name="Human D")
    plot_single(pt, ["Human - Human"], ax[2,0], name="Human")
    plot_single(pt, ["Human - Quiescence/Activation", "Human - Activation", "Human/Mouse - Activation"],
                ax[2,1], name="Human Q/A+A")
