import pandas as pd
import numpy as np
import cv2
import scipy.sparse as sparse

from ..image.utils import claher
from ..utils.parameters import RESOLVE_VOXEL_SIZE

##############################
### Helpers
##############################

def gene_to_upper(gene):
    """ Gene to upper case, adds _M for mouse genes.
    """
    if "~" in gene:
        s, l = gene.split(" ~ ")
        return gene_to_upper(s) + " ~ " + gene_to_upper(l)
    else:
        if gene == gene.upper() and gene.upper() not in ["H2-D1", "H2-K1", "PDGFRA", "MARCH4"]:
            return gene
        else:
            if gene == "mCherry":
                return gene.upper()
            elif gene == "eGFP":
                return "GFP"
            elif gene == "MARCH4":
                return "MARCHF4"
            else:
                return gene.upper()+"_M"

def read_Resolve_count(filepath, isbaysor=False):
    """ Read resolve count table.
    """
    if not isbaysor:
        return pd.read_table(filepath, header=None, names=["x","y","z","GeneR"], usecols=list(range(4)))
    else:
        counts = pd.read_table(filepath, sep=",")
        counts = counts.rename(columns={"gene":"GeneR"})
        return counts

def read_genemeta_file(filepath):
    """ Read gene meta file.
    """
    if ".xlsx" in filepath:
        genes = pd.read_excel(filepath).fillna("")
        genes.index = [gene.upper() if sp!="Mouse" else gene.upper()+"_M" for gene, sp in zip(genes["Gene"], genes["Species"])]
    else:
        genes = pd.read_table(filepath, sep=",", index_col=0).fillna("")
    return genes

##############################
### Class to load Resolve Data
##############################

class ResolveImage:
    """ Document me!
    """
    def __init__(self, filepath, imagepaths = {}, voxelsize = RESOLVE_VOXEL_SIZE, dosparse=False, isbaysor=False):
        self.voxelsize = voxelsize
        self.full_data = read_Resolve_count(filepath, isbaysor)
        if not isbaysor:
            self.full_data["GeneR"] = [gene_to_upper(g) for g in self.full_data["GeneR"]]
        else:
            self.full_data[["z","y","x"]] = np.round(self.full_data[["z","y","x"]]/self.voxelsize[:3],0).astype(int)
        
        genes = np.asarray(np.unique(self.full_data["GeneR"],return_counts=True)).T
        self.genes = pd.DataFrame(genes,columns=["GeneR","Count"]).sort_values(["Count"],
                                                                               ascending=False).reset_index(drop=True)
        self.genes.index = np.asarray(self.genes["GeneR"])
        self.gene_names = np.asarray(self.genes["GeneR"])
        
        self.imagesize = (100000,100000)
        
        self.images = {}
        for image in imagepaths:
            self.load_image(imagepaths[image], image)
            self.imagesize = self.images[image].shape
        
        single_gene_df = lambda gene: self.full_data[self.full_data["GeneR"]==gene].copy()
        filter_sg_df = lambda df: df.drop(columns=["GeneR"]).reset_index(drop=True)
        self.data = {gene: filter_sg_df(single_gene_df(gene)) for gene in self.gene_names}
        if dosparse: self.data2d = {gene: self.df_to_sparse_2D(self.data[gene]) for gene in self.data}
    
    def __getitem__(self, gene):
        return self.data2d[gene]
    
    def load_image(self, path, key, clip=True):
        self.images[key] = claher(cv2.imread(path, cv2.IMREAD_ANYDEPTH))
    
    def df_to_sparse_2D(self, df):
        return sparse.lil_matrix(sparse.coo_matrix((np.ones_like(df["x"]), (df["y"],df["x"])), shape=self.imagesize))
    
    def add_metadata(self, filepath):
        genes = read_genemeta_file(filepath)
        #genes["GeneMod"] = [gene.upper() if sp!="Mouse" else gene.upper()+"_M" for gene, sp in zip(genes["Gene"], genes["Species"])]
        #genes.index = np.asarray(genes["GeneMod"])
        self.genes = pd.merge(self.genes,genes,left_index=True,right_index=True,how="left").sort_values("Count",ascending=False)
