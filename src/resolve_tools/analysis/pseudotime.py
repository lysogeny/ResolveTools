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
    
def get_human_glmpca_data(adata, N=2):
    """ Reduce full adata to human cells for pseudotime estimation.
    """
    adatapca = adata[np.logical_and(adata.obs["HumanGeneCount"]>N, adata.obs["IS_HUMAN"])]
    adatapca = adatapca[:,adatapca.var["Species"]!="Mouse"].copy()
    counts = np.asarray(adatapca.X.todense() if not (type(adata.X)==np.matrix or type(adata.X)==np.ndarray) else adata.X).T #[adatapca.var["Species"]!="Mouse"]

    df = adatapca.to_df().T
    df.index = adatapca.var["GeneClass"]
    df = df.groupby("GeneClass").apply(lambda x: x.sum(axis=0))
    df = df.T
    adatapcashort = anndata.AnnData(df, obs=adatapca.obs).copy()
    adatapcashort.var["Gene"] = list(adatapcashort.var.index)
    adatapcashort = adatapcashort[adatapcashort.X.sum(axis=1)>0].copy()
    countsshort = np.asarray(adatapcashort.X).T #[adatapcashort.var["Gene"].apply(lambda x: "Human" in x)]
    adatapcashort.X = adatapcashort.X/adatapcashort.X.sum(axis=1)[:,None]
    adatapca.X = adatapca.X/adatapca.X.sum(axis=1)[:,None]
    return adatapca, countsshort, adatapcashort

##############################
### PT
##############################

def get_pseudotime(countsshort, adatapcashort, N=2, verbose=False, separation=0.3, tryFull=False, **kwargs):
    """ Get PT using glmpca.
    """
    printwtime("    Run glm-pca")
    factorsshort = glmpca.glmpca(countsshort, N, fam="nb", verbose=verbose, **kwargs)
    ctypesordered = ['Human Q', 'Human Q/A', 'Human A', 'Human D']
    def get_single_PT(i):
        pt = np.argsort(np.argsort(factorsshort["factors"][:,i]))
        pt = pt/pt.max()-pt.min()
        def meanstd(x):
            return x.mean(), x.std()
        means = np.asarray([meanstd(pt[adatapcashort.obs["BaysorClusterCelltype"]==ct]) for ct in ctypesordered])
        return means, pt, factorsshort
    for i in ([0] if not tryFull else range(N)):
        means, pt, factorsshort = get_single_PT(i)
        diff = (means[1:,0]-means[:-1,0])/np.sqrt(means[1:,1]**2+means[:-1,1]**2)
        printwtime("    "+str(i)+" "+str(diff))
        if np.all(diff<-separation):
            return True, 1-pt, i, factorsshort
        if np.all(diff>separation):
            return True, pt, i, factorsshort
    return False, np.zeros(pt.shape), -1, factorsshort

def get_meanpseudotime(countsshort, adatapcashort, meanN=10, N=2, verbose=False, separation=0.3, tryFull=False, maxN=100, **kwargs):
    """ Get PT multiple times, final PT from mean.
    """
    finalptlist = list()
    finalptindex = list()
    c = 0
    while len(finalptlist)<meanN and c<maxN:
        printwtime("Run "+str(c))
        c += 1
        
        try:
            success, pt, i, _ = get_pseudotime(countsshort, adatapcashort, N=N, verbose=verbose, separation=separation, tryFull=tryFull, **kwargs)
        except glmpca.GlmpcaError:
            success = False
            printwtime("  GlmpcaError!")
        
        if success:
            finalptlist.append(pt)
            finalptindex.append(i)
            printwtime("  Found PT in component "+str(i))
        else:
            printwtime("  Found no PT")
    
    if len(finalptlist)!=meanN:
        printwtime("  Could not find enough PT!")
        assert False
    
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
        smoother = loess(pt, yvals, span=0.05, degree=1)
        xout = np.arange(0,1.001,0.001)
        yout = smoother.predict(xout).values
        sns.lineplot(x=xout, y=yout, ax=ax, color="black")
        ax.set_title(groups[0] if not name else name)
        ax.set_ylim([-0.03,1.03])

    fig, ax = plt.subplots(3,2,figsize=(20,20))
    plot_single(pt, ["Human - Quiescence"], ax[0,0], name="Human Q")
    plot_single(pt, ["Human - Quiescence/Activation"], ax[0,1], name="Human Q/A")
    plot_single(pt, ["Human - Activation", "Human/Mouse - Activation"], ax[1,0], name="Human A")
    plot_single(pt, ["Human - Differentiation"], ax[1,1], name="Human D")
    plot_single(pt, ["Human - Human"], ax[2,0], name="Human")
    plot_single(pt, ["Human - Quiescence/Activation", "Human - Activation", "Human/Mouse - Activation"],
                ax[2,1], name="Human Q/A+A")
