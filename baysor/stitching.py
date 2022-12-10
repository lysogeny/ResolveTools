import scipy.spatial as spatial
import itertools
import pandas as pd
import numpy as np

class BaysorCell:
    """ Segmentated cell from Baysor.
    """
    def __init__(self, series):
        #self.brainregion = series["BrainRegion"]
        #self.brainregionname = series["BrainRegionName"]
        self.name = series["CellName"]
        self.index = series["MaskIndex"]
        self.roi = series["ROI"]
        #self.position = np.asarray(series[["x","y","z"]])
        self.cluster = series["cluster"]
        self.transcriptcount = series["n_transcripts"]
        self.segcell = 0
        
        self.tpositions = np.zeros((0))
        self.tclusters = np.zeros((0))
        self.direct_connections = np.zeros((2,0))
        self.baysorneighbors = []
        
    def __repr__(self):
        text = "Baysor cell "+self.name+" with index "+str(self.index)+" and cluster "+str(self.cluster)+"."
        text += "\nCurrently assigned to segmentation cell "+str(self.segcell)+"."
        return text
    
    def set_position_from_transcripts(self):
        """ Set centroid from assigned transcripts.
        """
        self.position = self.tpositions.mean(axis=0)
    
    def process_direct_connections(self):
        """ Process direct_connections after it was set.
        """
        self.direct_connections_count = self.direct_connections[1][np.argsort(self.direct_connections[1])[::-1]]
        self.direct_connections_index = self.direct_connections[0][np.argsort(self.direct_connections[1])[::-1]]
        self.try_direct_connection = len(self.direct_connections_index)>0
        self.direct_connection_was_resolved = False
    
    def choose_direct_connection(self, i):
        """ Choose direct connection i from available.
        """
        if len(self.direct_connections_count)>1:
            self.direct_connections = np.asarray([self.direct_connections[0][i:i+1], self.direct_connections[1][i:i+1]])
            self.direct_connections_count = self.direct_connections_count[i:i+1]
            self.direct_connections_index = self.direct_connections_index[i:i+1]
            self.direct_connection_was_resolved = True
    
    def clean_baysor_neighbors(self):
        """ Exclude self from Baysor neighbor list.
        """
        if self.index in self.baysorneighbors:
            self.baysorneighbors = self.baysorneighbors.remove(self.index)


class SegmentationCell:
    """ Segmentated cell from the DAPI image.
    """
    #__slots__ = ("brainregion","brainregionname","name","index","roi","position","cluster","baysorcells")
    def __init__(self, series):
        self.brainregion = series["BrainRegion"]
        self.brainregionname = series["BrainRegionName"]
        self.name = series["CellName"]
        self.index = series["MaskIndex"]
        self.roi = series["ROI"]
        self.position = np.asarray(series[["x","y","z"]]).astype(float)
        self.cluster = 0
        self.baysorcells = []
        
    def __repr__(self):
        text = "Segmentation cell "+self.name+" with index "+str(self.index)+" in region "+self.brainregionname+"."
        text += "\nCurrently assigned to Baysor cluster "+str(self.cluster)
        #text += " with "+str(len(self.baysorcells))+" assigned Baysor cells."
        text += " with assigned Baysor cells "+str(self.baysorcells)+"."
        return text


@pd.api.extensions.register_series_accessor("cell")
class CellAccessor:
    """ Accessor extension for pandas Series.
        Adds namespace .cell to all Series.
        Access properties of cells in a Series with e.g.: col.cell.cget("property")
        Set properties of cells in a Series with e.g.: col.cell.cset("property", data)
        
        Be careful when setting values! I probably use less checks than pandas would use
        if "property" was just a DataFrame column. So be sure it really does what you want!
    """
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass
    
    def cget(self, name):
        """ Get attribute name from all cells, return as series
        """
        return self._obj.apply(lambda x: getattr(x, name))
    
    def cset(self, name, data):
        """ Set attribute name for all cells to corresponding value in series.
            data can either be a list/array, or a pd.Series with correct index or a single value.
        """
        try:
            if type(data) is pd.Series:
                self._obj.apply(lambda x: setattr(x, name, data[x.index]))
            elif len(data) == len(self._obj):
                for x, val in zip(self._obj, data):
                    setattr(x, name, val)
            else:
                self._obj.apply(lambda x: setattr(x, name, data))
        except TypeError:
            self._obj.apply(lambda x: setattr(x, name, data))


class SegmentedResolveROI:
    """ 
    """
    def __init__(self, resultfolder, segloomfile, genemetafile, verbose=True):
        self.verbose = verbose
        if self.verbose: printwtime(f"Reading initial ROI data")
        self.adata_segmentation = read_loom(segloomfile)
        self.adata_segmentation.obs.index = self.adata_segmentation.obs["MaskIndex"]
        self.obsseg = self.adata_segmentation.obs
        self.cellsseg = self.obsseg.apply(lambda x: SegmentationCell(x), axis=1)
        
        assert len(np.unique(self.adata_segmentation.obs["ROI"]))==1
        self.roi = np.unique(self.adata_segmentation.obs["ROI"])[0]
        
        self.adata_baysor = assign_counts_from_Baysor(resultfolder, genemetafile, self.roi)
        self.adata_baysor.obs.index = self.adata_baysor.obs["MaskIndex"]
        self.obsbay = self.adata_baysor.obs
        self.cellsbay = self.obsbay.apply(lambda x: BaysorCell(x), axis=1)
        
        self.transcripts_wnoise = pd.read_table(resultfolder+"/segmentation.csv", sep=",")
        self.transcripts_wnoise["celltype"] = np.asarray((self.adata_baysor.var["Species"] + " - " + \
                                              self.adata_baysor.var["Celltype"]).loc[self.transcripts_wnoise["gene"]])
        self.transcripts = self.transcripts_wnoise[~self.transcripts_wnoise["is_noise"]].copy().reset_index(drop=True)

    def __repr__(self):
        text = f"Data for ROI {self.roi}."
        text += f"\nContains {len(self.cellsseg)} segmentation cells, {len(self.cellsbay)} Baysor cells."
        share = int(np.round(1-len(self.transcripts)/len(self.transcripts_wnoise),2)*100)
        text += f"\nContains {len(self.transcripts)} transcripts that are not noise, the noise share is {share}%."
        return text
    
    def _cgetbay(self, name):
        return self.cellsbay.cell.cget(name)
    def _csetbay(self, name, data):
        return self.cellsbay.cell.cset(name, data)
    def _cgetseg(self, name):
        return self.cellsseg.cell.cget(name)
    def _csetseg(self, name, data):
        return self.cellsseg.cell.cset(name, data)
    def _applybay(self, function):
        self.cellsbay.apply(function)
    def _applyseg(self, function):
        self.cellsseg.apply(function)
    
    def cluster_crosstab(self, norm=True, normgenes=True, wnoise=True):
        """ Crosstab of Baysor transcripts with assigned cluster.
        """
        trans = self.transcripts_wnoise if wnoise else self.transcripts
        cross = pd.crosstab(trans["celltype"], trans["cluster"])
        if not norm: return cross
        if normgenes:
            cross = np.round(cross/np.asarray(cross.sum(axis=1))[:,None]*100,0).astype(int)
            cross = cross.astype(str)
            cross[cross=="0"] = ""
            return cross
        else:
            cross = np.round(cross/np.asarray(cross.sum(axis=0))[None]*100,0).astype(int)
            cross = cross.astype(str)
            cross[cross=="0"] = ""
            return cross
    
    def combine_clusters(self, clusterlist=[]):
        for pair in clusterlist:
            self.cellsbay[self.cellsbay.cell.cget("cluster")==pair[0]].cell.cset("cluster", pair[1])
            self.transcripts.loc[self.transcripts["cluster"]==pair[0], "cluster"] = pair[1]
            self.transcripts_wnoise.loc[self.transcripts_wnoise["cluster"]==pair[0], "cluster"] = pair[1]
    
    def add_baysor_initial(self):
        """ Add initial Baysor cell properties from transcript list.
        """
        if self.verbose: printwtime(f"Setting initial Baysor cell properties from transcripts")
        groupedtranscripts = self.transcripts.groupby(["cell"])
        # Cluster identity for all contained transcripts
        self._csetbay("tclusters", groupedtranscripts.apply(lambda x: np.asarray(x["cluster"])))
        # Position for all contained transcripts, zyx
        self._csetbay("tpositions", groupedtranscripts.apply(lambda x: np.asarray(x[["x","y","z"]])))
        # Center of mass of all contained transcripts
        self.cellsbay.apply(lambda x: x.set_position_from_transcripts())
    
    def add_direct_connectivity(self):
        """ Add direct connectivity to Baysor cells.
        """
        if self.verbose: printwtime(f"Adding direct connectivity to Baysor cell properties")
        groupedtranscripts = self.transcripts.groupby(["cell"])
        def get_prior_segmentations(df):
            u, c = np.unique(df["prior_segmentation"], return_counts=True)
            return [u[u!=0], c[u!=0]]
        # Add prior connections of Baysor cells
        prior = groupedtranscripts.apply(get_prior_segmentations)
        self._csetbay("direct_connections_full", prior)
        self._csetbay("direct_connections", prior)
        # Process them (mostly sorting)
        self.cellsbay.apply(lambda x: x.process_direct_connections())
        
    def resolve_lopsided_nonunique_direct_connections(self, resolvemultipleabove=0.82):
        """ For cells that have multiple direct connections to segmentation cells, resolve
            them to that with the highest overlap if the one with the highest has at least
            resolvemultipleabove share of the total overlap counts. 0.82 corresponds to
            at least a ratio 5:1.
        """
        if self.verbose: printwtime(f"Resolve lopsided non-unique direct connections")
        # Find lopsided connections
        nonunique = self.cellsbay[self._cgetbay("direct_connections_index").apply(len)>1]
        decide = nonunique[nonunique.cell.cget("direct_connections_count").apply(lambda x: x[0]/x.sum())>resolvemultipleabove]
        # Tell cells to resolve to the first connection (that with highest overlap) for those connections
        decide.apply(lambda x: x.choose_direct_connection(0))
        if self.verbose: printwtime(f"  Decided on unique connection from lopsidedness with cutoff {resolvemultipleabove} for {len(decide)} Baysor cells, {len(nonunique)-len(decide)} non-unique left")
    
    def resolve_nonunique_from_centroid_distance(self):
        """ Resolve remaining non-unique connections from simple centroid distance.
            
            IMPROVE THIS TO A BETTER METHOD!!!!!
        """
        if self.verbose: printwtime(f"Resolve remaining non-unique direct connections")
        # Find closest connection for cells
        nonunique = self.cellsbay[self._cgetbay("direct_connections_index").apply(len)>1]
        def get_distances(cell):
            own = cell.position
            con = np.asarray(list(self.cellsseg.loc[cell.direct_connections_index].apply(lambda x: x.position.astype(float))))
            return np.sqrt(((own[None]-con)**2).sum(axis=1))
        closest = nonunique.apply(get_distances).apply(np.argmin)
        # Resolve to closest connection
        for cell, close in zip(nonunique, closest):
            cell.choose_direct_connection(close)
        if self.verbose: printwtime(f"  Decided on unique connection from simple distance for {len(decide)} Baysor cells. IMPROVE THIS!!!!!")
    
    def add_baysor_neighbors(self, trsconnsearchdistance = 10):
        """ Find all neighbors within Baysor cells with min distance
            of any transcripts trsconnsearchdistance.
        """
        if self.verbose: printwtime(f"Find all Baysor cell neighbors of Baysor cells")
        # Find point neighbors
        points = np.asarray(self.transcripts[["x","y","z"]])
        point_tree = spatial.cKDTree(points)
        self.transcripts["neighbor_transcripts"] = point_tree.query_ball_point(points, trsconnsearchdistance, workers=-1)
        # Find cell neighbors
        groupedtranscripts = self.transcripts.groupby(["cell"])
        def get_cells(x):
            transcripts = np.unique(list(itertools.chain.from_iterable(x["neighbor_transcripts"])))
            cells = list(np.unique(self.transcripts.loc[transcripts,"cell"]))
            return cells
        neighbors = groupedtranscripts.apply(get_cells)
        # Add to cells and clean self
        self._csetbay("baysorneighbors", neighbors)
        self.cellsbay.apply(lambda x: x.clean_baysor_neighbors())
        
        # Add distance for all neighbors to Baysor cells
        def cell_trsdistance(i,j):
            posi, posj = self.cellsbay.loc[i].tpositions, self.cellsbay.loc[j].tpositions
            dist = spatial.distance_matrix(posi, posj)
            return dist.mean(), dist.min(), dist.max()
        def dst_alln(cell):
            if len(cell.baysorneighbors)>0:
                dist = np.asarray([cell_trsdistance(cell.index,i) for i in cell.baysorneighbors])
                cell.baysorneighbors_distance_mean = dist[:,0]
                cell.baysorneighbors_distance_min = dist[:,1]
                cell.baysorneighbors_distance_max = dist[:,2]
            else:
                cell.baysorneighbors_distance_mean = []
                cell.baysorneighbors_distance_min = []
                cell.baysorneighbors_distance_max = []
        self.cellsbay.apply(lambda x: dst_alln(x))

