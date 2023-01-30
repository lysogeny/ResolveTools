import scipy.spatial as spatial
import itertools
import pandas as pd
import numpy as np
import anndata
import warnings
import matplotlib.pyplot as plt

from .cell import BaysorCell, SegmentationCell
from .utils import assign_counts_from_Baysor
from ..segmentation.counts import read_loom
from ..utils.utils import printwtime
from ..resolve.resolveimage import read_genemeta_file
from .visualization import plot_final_assignment


##############################
### Custom Cell Accessor
##############################

@pd.api.extensions.register_series_accessor("cell")
class CellAccessor:
    """ Accessor extension for pandas Series.
        Adds namespace .cell to all Series.
        
        Access properties of cells in a Series with: col.cell.cget("property")
        Set properties of cells in a Series with: col.cell.cset("property", data)
        
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
            
            .copy() is important, otherwise the value is set by reference!
        """
        try:
            if type(data) is pd.Series:
                self._obj.apply(lambda x: setattr(x, name, data[x.index]))
            elif len(data) == len(self._obj):
                for x, val in zip(self._obj, data):
                    setattr(x, name, val)
            else:
                self._obj.apply(lambda x: setattr(x, name, data.copy()))
        except TypeError:
            self._obj.apply(lambda x: setattr(x, name, data.copy()))

##############################
### Full ROI
##############################

class SegmentedResolveROI:
    """ 
        
        Still has some discreptancy when reducing nonunique to my original code??!
        
        Make sure to use enough .copy() statements!
    """
    def __init__(self, resultfolder, segloomfile, genemetafile, verbose = True):
        self.verbose = verbose
        if self.verbose: printwtime(f"Reading initial ROI data")
        self.adata_segmentation = read_loom(segloomfile)
        self.adata_segmentation.obs.index = self.adata_segmentation.obs["MaskIndex"]
        self.obsseg = self.adata_segmentation.obs
        self.cellsseg = self.obsseg.apply(lambda x: SegmentationCell(x), axis=1)
        
        if len(np.unique(self.adata_segmentation.obs["ROI"]))!=1:
            raise ValueError("Data does not only contain a single ROI! Not implemented yet.")
        self.roi = np.unique(self.adata_segmentation.obs["ROI"])[0]
        
        self.adata_baysor = assign_counts_from_Baysor(resultfolder, genemetafile, self.roi)
        self.adata_baysor.obs.index = self.adata_baysor.obs["MaskIndex"]
        self.obsbay = self.adata_baysor.obs
        self.cellsbay = self.obsbay.apply(lambda x: BaysorCell(x), axis=1)
        
        self.genemeta = read_genemeta_file(genemetafile)
        self.transcripts_wnoise = pd.read_table(resultfolder+"/segmentation.csv", sep=",")
        if "prior_segmentation" not in self.transcripts_wnoise.columns:
            self.transcripts_wnoise["prior_segmentation"] = 0
        self.transcripts_wnoise["celltype"] = np.asarray((self.genemeta["Species"] + " - " + \
                                              self.genemeta["Celltype"]).loc[self.transcripts_wnoise["gene"]])
        self.transcripts_wnoise["celltypegene"] = self.transcripts_wnoise["celltype"] + " - " + self.transcripts_wnoise["gene"]
        self.transcripts = self.transcripts_wnoise[~self.transcripts_wnoise["is_noise"]].copy().reset_index(drop=True)
    
    ##############################
    ### Utility functions
    ##############################
    
    def __repr__(self):
        text = f"Data for ROI {self.roi}."
        text += f"\nContains {len(self.cellsseg)} segmentation cells, {len(self.cellsbay)} Baysor cells."
        share = int(np.round(1-len(self.transcripts)/len(self.transcripts_wnoise),2)*100)
        Ncluster = np.unique(self.transcripts["cluster"]).shape[0]
        text += f"\nContains {len(self.transcripts)} transcripts that are not noise in {Ncluster} clusters, the noise share is {share}%."
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
        
    ##############################
    ### Cluster Processing
    ##############################
    
    def cluster_crosstab(self, norm = True, normgenes = True, wnoise = True, comparekey = "celltype"):
        """ Crosstab of Baysor transcripts with assigned cluster.
        """
        trans = self.transcripts_wnoise if wnoise else self.transcripts
        cross = pd.crosstab(trans[comparekey], trans["cluster"])
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
    
    def cluster_combine(self, clusterlist=[]):
        """ Combine clusters. Should be done before adding any of the initial.
            Order of the combinations in clusterlist is irrelevant, will
            always reduce all equivalent clusters to that with the lowest index.
        """
        Ninit = np.unique(self.transcripts["cluster"]).shape[0]
        clusterlistsorted = [sorted(l, reverse=True) for l in sorted(clusterlist, key=max, reverse=True)]
        replacedict = dict(zip(np.arange(self.transcripts["cluster"].max()+1),np.arange(self.transcripts["cluster"].max()+1)))
        for repl_ in clusterlistsorted:
            repl = [replacedict[k] for k in repl_]
            for key in replacedict:
                if replacedict[key] in repl:
                    replacedict[key] = min(repl)
        for key in replacedict:
            self.cellsbay[self.cellsbay.cell.cget("cluster")==key].cell.cset("cluster", replacedict[key])
            self.transcripts.loc[self.transcripts["cluster"]==key, "cluster"] = replacedict[key]
            self.transcripts_wnoise.loc[self.transcripts_wnoise["cluster"]==key, "cluster"] = replacedict[key]
            self.obsbay.loc[self.obsbay["cluster"]==key, "cluster"] = replacedict[key]
        Nfinal = np.unique(self.transcripts["cluster"]).shape[0]
        if self.verbose: printwtime(f"Initially contained {Ninit} distinct clusters, got reduced to {Nfinal}.")
    
    ##############################
    ### Add Initial Data
    ##############################
    
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
        # Prior segmentation
        self._csetbay("tprior", groupedtranscripts.apply(lambda x: np.asarray(x["prior_segmentation"])))
        
        # Add transcript counts
        def set_counts(x):
            self.cellsbay.loc[x.name].counts = x.copy()
        self.adata_baysor.to_df().apply(set_counts, axis=1)
    
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
        
        if self.verbose: printwtime(f"  Add Baysor neighbor distances")
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
        self.cellsbay.apply(lambda x: dst_alln(x))
    
    def add_baysor_segneighbors(self, boundaryfile = "", maxsearchdistseg = 20):
        """ Find all segmentation cell neighbors of the Baysor cells within
            maxsearchdistseg distance of the centroids.
            Doesn't find some edge cases with very large Baysor clusters
            (diameter>maxsearchdistseg), but those are bad anyway.
        """
        if self.verbose and not boundaryfile:
            printwtime(f"No boundary for segmentation cells was provided, skipping neighbor search.")
        # First find potential neighbors for all cells
        if self.verbose: printwtime(f"Find all segmentation cell neighbors of Baysor cells")
        index = self._cgetseg("index").copy().reset_index(drop=True)
        pointsseg = np.asarray(list(self._cgetseg("position")))
        point_tree = spatial.cKDTree(pointsseg)
        pointsbay = np.asarray(list(self._cgetbay("position")))
        neighbors = pd.Series(point_tree.query_ball_point(pointsbay, maxsearchdistseg, workers=-1))
        neighbors = neighbors.apply(lambda x: np.asarray(index.loc[x]))
        self._csetbay("segneighbors", list(neighbors))
        
        # Now get approximate distance to boundary for all transcripts
        if self.verbose: printwtime(f"  Add Baysor segmentation neighbor distances")
        boundary = pd.Series(np.load(boundaryfile, allow_pickle=True)["points"], index=np.load(boundaryfile)["index"])
        def add_distances(cell):
            if len(cell.segneighbors)>0:
                dists = np.asarray([spatial.distance_matrix(cell.tpositions, boundary.loc[neighbor]).min(axis=1)
                                    for neighbor in cell.segneighbors])
                dists[cell.tprior[None,:] == np.asarray(cell.segneighbors)[:,None]] = 0
                cell.segneighbors_distance_mean = dists.mean(axis=1)
                cell.segneighbors_distance_min = dists.min(axis=1)
                cell.segneighbors_distance_max = dists.max(axis=1)
            mask = [x in list(cell.segneighbors) for x in cell.direct_connections_index]
            cell.direct_connections_index = np.asarray(cell.direct_connections_index)[mask]
            cell.direct_connections_count = np.asarray(cell.direct_connections_count)[mask]
        self.cellsbay.apply(lambda x: add_distances(x))
    
    def add_initial(self, boundaryfile = "", trsconnsearchdistance = 10, maxsearchdistseg = 20):
        """ Add all initial data.
        """
        self.add_baysor_initial()
        self.add_direct_connectivity()
        self.add_baysor_neighbors(trsconnsearchdistance)
        add_baysor_segneighbors(self, boundaryfile, maxsearchdistseg)
    
    ##############################
    ### Resolve non-unique direct connections
    ##############################
    
    def resolve_lopsided_nonunique_direct_connections(self, resolvemultipleabove = 0.82):
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
            
            resolve_nonunique_from_surface_distance should probably be the preferred method.
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
        if self.verbose: printwtime(f"  Decided on unique connection from simple centroid distance for {len(nonunique)} Baysor cells.")
    
    def resolve_nonunique_from_surface_distance(self):
        """ Resolve remaining non-unique connections from mean distance of
            transcripts to the surface of the segmentation cell.
        """
        if self.verbose: printwtime(f"Resolve remaining non-unique direct connections")
        # Find closest connection for cells
        nonunique = self.cellsbay[self._cgetbay("direct_connections_index").apply(len)>1]
        closest = nonunique.apply(lambda x: [x.segneighbors_distance_mean[list(x.segneighbors).index(i)]
                                             for i in x.direct_connections_index]).apply(np.argmin)
        for cell, close in zip(nonunique, closest):
            cell.choose_direct_connection(close)
        if self.verbose: printwtime(f"  Decided on unique connection from mean distance to segmentation surface for {len(nonunique)} Baysor cells.")
    
    def resolve_nonunique_direct(self, resolvemultipleabove = 0.82):
        """ Resolve all non-unique direct connections from Baysor cells
            to segmentation cells.
        """
        self.resolve_lopsided_nonunique_direct_connections(0.82)
        self.resolve_nonunique_from_surface_distance()
    
    ##############################
    ### Assign baysor clusters to segmentation cells
    ##############################
    
    def assign_baysor_toseg_direct(self, ambigN = 10, ambigShare = 0.6):
        """ Assign Baysor cells that are obvious, i.e. where exactly one Baysor cell has a
            transcript intersection with exactly one segmentation cell, or where multiple cells
            from the same cluster uniquely map to a single segmentation cell.
            
            Also assigns to segmentation cells where the clusters mapping to it have at least ambigN
            total counts, and ambigShare of them share the same cluster. Drops the Baysor clusters
            from direct assignment that don't share this cluster.
            
            Assigns cluster identities to the relevant segmentation cells.
        """
        if self.verbose: printwtime(f"Do simple cell assignment for {len(self.cellsbay)} Baysor cells and {len(self.cellsseg)} segmentation cells")
        totalassigned = 0
        
        candidates = self._cgetbay("direct_connections_index")[self._cgetbay("direct_connections_index").apply(len)>0]
        if self.verbose: printwtime(f"  Contains {len(candidates)} Baysor cells that have any direct connection to the segmentation")
        if self.verbose and len(candidates)==0:
            printwtime(f"  Found no direct connections, skipping assignment from direct connections")
            return None
        
        # First assign cells to segmentation where the connection is unique in both directions
        u, c = np.unique(list(itertools.chain.from_iterable(candidates)), return_counts=True)
        mapto = pd.DataFrame(np.asarray([u,c]).T, columns=["index","count"])
        mapto.index = mapto["index"]
        unique = set(u[c==1])
        
        assign = candidates[candidates.apply(len)==1].apply(lambda x: x[0])
        assign = assign[assign.apply(lambda x: x in unique)]
        for cbay, cseg in assign.items():
            self.cellsbay.loc[cbay].assign_to_segcell(cseg)
            self.cellsseg.loc[cseg].cluster = self.cellsbay.loc[cbay].cluster
        totalassigned += len(assign)
        if self.verbose: printwtime(f"    Assigned {len(assign)} Baysor cells where the direct connection is mutually unique")
        
        # Find cells where the connection from cell to segmentation cell is clear
        # And only include those where no unclear cells map to a segmentation cell
        candidates = self._cgetbay("direct_connections_index")[self._cgetbay("try_direct_connection")]
        maybe = candidates[candidates.apply(len)==1].apply(lambda x: x[0])
        maybe = maybe[maybe.apply(lambda x: x not in unique)]
        
        u, c = np.unique(maybe, return_counts=True)
        mapto = mapto.loc[u]
        onewayclear = set(mapto[mapto["count"]==c]["index"])
        maybe = maybe[maybe.apply(lambda x: x in onewayclear)]
        if self.verbose: printwtime(f"  Contains {len(maybe)} Baysor cells with one-way unique mappings")
        
        # Map those where mapping multiple cells to a segmentation is possible without cluster collisions
        df = self.obsbay.loc[maybe.index, ["cluster","n_transcripts"]].copy()
        df["connection"] = maybe
        con = df.groupby(["connection"]).apply(lambda x: len(np.unique(x["cluster"]))==1)
        
        nocollisions = set(con[con].index)
        assign = maybe[maybe.apply(lambda x: x in nocollisions)]
        for cbay, cseg in assign.items():
            self.cellsbay.loc[cbay].assign_to_segcell(cseg)
            self.cellsseg.loc[cseg].cluster = self.cellsbay.loc[cbay].cluster
        totalassigned += len(assign)
        if self.verbose: printwtime(f"    Assigned {len(assign)} Baysor cells where this causes no cluster collisions")
        
        # Now try those with collisions
        ambiguous = maybe[maybe.apply(lambda x: x not in nocollisions)]
        df = df.loc[ambiguous.index]#df[~df.index.isin(assign.index)]
        df["Tclusters"] = self.cellsbay.loc[ambiguous.index].cell.cget("tclusters")
        
        # Take all segmentation cells where the clusters mapping to it have at least ambigN counts,
        # and ambigShare of them share the same cluster
        celltrs = df.groupby(["connection"]).apply(lambda x: list(itertools.chain.from_iterable(x["Tclusters"])))
        def get_shares(x):
            u, c = np.unique(x, return_counts=True)
            ind = np.argsort(c)[::-1]
            return u[ind], np.round(c[ind]/c.sum(),2), c.sum()
        celltrs = celltrs.apply(get_shares)
        celltrs = celltrs[celltrs.apply(lambda x: x[1][0]>=ambigShare and x[2]>=ambigN)].apply(lambda x: x[0][0])
        
        # Apply that cluster to the segmentation cells, and add the cells that also belong to this cluster
        self.cellsseg.loc[celltrs.index].cell.cset("cluster",celltrs)
        df["clusterSeg"] = list(self.cellsseg.loc[df["connection"]].cell.cget("cluster"))
        df = df[df["clusterSeg"]!=0]
        df = df[df["cluster"]==df["clusterSeg"]]
        assign = df["connection"]
        
        for cbay, cseg in assign.items():
            self.cellsbay.loc[cbay].assign_to_segcell(cseg)
        totalassigned += len(assign)
        if self.verbose: printwtime(f"    Assigned {len(assign)} conflicting Baysor cells, using N={ambigN}, share={ambigShare} as cutoff")
        leftover = ambiguous.loc[~ambiguous.index.isin(assign.index)]
        for cbay, cseg in leftover.items():
            self.cellsbay.loc[cbay].assign_to_segcell(0)
        if self.verbose: printwtime(f"    Dropped {len(leftover)} Baysor cells with a direct connection from direct assignment")
        
        # Print some final statistics
        left = self._cgetbay("try_direct_connection").sum()
        if self.verbose: printwtime(f"  Assigned {totalassigned} cells, dropped {len(leftover)} cells, {left} cells with relevant direct connections remain")
        segass = (self._cgetseg("cluster")!=0).sum()
        if self.verbose: printwtime(f"  Assigned counts to {segass} segmentation cells")
        tat = self._cgetbay("transcriptcount")[self._cgetbay("segcell")!=0].sum()
        tuct = self._cgetbay("transcriptcount")[self._cgetbay("try_direct_connection")].sum()
        tot = self._cgetbay("transcriptcount").sum() -tat -tuct
        if self.verbose: printwtime(f"Overall assigned {tat} transcripts, {tuct} remain with rel. direct connection, {tot} in other cells")
    
    def assign_do_single_tryassign_fromdistance(self, unassigned, ambigN = 8, ambigShare = 0.55):
        """ Try a single round of assigning unassigned Baysor cells to segcells.
        """
        maybe = unassigned.apply(lambda x: x.segneighbors[np.argmin(x.segneighbors_distance_mean)])

        # Map those where mapping multiple cells to a segmentation is possible without cluster collisions
        # No need to look as segcell cluster here, conflicting neighbors are already filtered
        df = self.obsbay.loc[maybe.index, ["cluster","n_transcripts"]].copy()
        df["connection"] = maybe
        df["distance"] = unassigned.apply(lambda x: np.min(x.segneighbors_distance_mean))
        con = df.groupby(["connection"]).apply(lambda x: len(np.unique(x["cluster"]))==1)

        nocollisions = set(con[con].index)
        assign = maybe[maybe.apply(lambda x: x in nocollisions)]
        for cbay, cseg in assign.items():
            self.cellsbay.loc[cbay].assign_to_segcell(cseg)
            self.cellsseg.loc[cseg].cluster = self.cellsbay.loc[cbay].cluster
        if self.verbose: printwtime(f"    Assigned {len(assign)} Baysor cells to {len(nocollisions)} segcells where this causes no cluster collisions")

        # Now try those with collisions
        # Only possible if segcell has no assignment yet
        ambiguous = maybe[maybe.apply(lambda x: x not in nocollisions)]
        df = df.loc[ambiguous.index]#df[~df.index.isin(assign.index)]
        df["Tclusters"] = self.cellsbay.loc[ambiguous.index].cell.cget("tclusters")
        df["Tmeandist"] = df.apply(lambda x: [x["distance"]]*x["n_transcripts"], axis=1)

        # Take all segmentation cells where the clusters mapping to it have at least ambigN counts,
        # and ambigShare of them share the same cluster; here ambigShare is weighted with the mean distance of the cluster.
        def get_shares(x, add=0.5):
            """ Get shares of clusters, with every transcript weighted by 1/(mean_cluster_distance + add).

                Could also use distance of every transcript, instead of cluster mean?!
            """
            clst = np.asarray(list(itertools.chain.from_iterable(x["Tclusters"])))
            dstwgt = 1/(np.asarray(list(itertools.chain.from_iterable(x["Tmeandist"])))+add)

            unq,inv = np.unique(clst,return_inverse=True)
            c = np.bincount(inv,dstwgt.reshape(-1))

            ind = np.argsort(c)[::-1]
            return unq[ind], np.round(c[ind]/c.sum(),2), inv.shape[0]
        celltrs = df.groupby(["connection"]).apply(lambda x: get_shares(x))
        celltrs = celltrs[celltrs.apply(lambda x: x[1][0]>=ambigShare and x[2]>=ambigN)].apply(lambda x: x[0][0])

        # Apply that cluster to the segmentation cells, and add the cells that also belong to this cluster
        self.cellsseg.loc[celltrs.index].cell.cset("cluster",celltrs)
        df["clusterSeg"] = list(self.cellsseg.loc[df["connection"]].cell.cget("cluster"))
        df = df[df["clusterSeg"]!=0]
        df = df[df["cluster"]==df["clusterSeg"]]
        assign = df["connection"]

        for cbay, cseg in assign.items():
            self.cellsbay.loc[cbay].assign_to_segcell(cseg)
        if self.verbose: printwtime(f"    Assigned {len(assign)} conflicting Baysor cells to {len(celltrs)} segcells, using N={ambigN}, weighted share={ambigShare} as cutoff")
    
    def assign_print_stats(self):
        """ Print some statistics.
        """
        nassigned = (self._cgetbay("segcell")!=0).sum()
        ntot = len(self.cellsbay)
        nassignedseg = (self._cgetseg("cluster")!=0).sum()
        ntotseg = len(self.cellsseg)
        tat = self._cgetbay("transcriptcount")[self._cgetbay("segcell")!=0].sum()
        tot = self._cgetbay("transcriptcount").sum() -tat
        if self.verbose: printwtime(f"Overall, ROI has {ntot} Baysor cells, of which {nassigned} are assigned to a segmentation cell.")
        if self.verbose: printwtime(f"Overall, ROI has {ntotseg} segmentation cells, of which {nassignedseg} are assigned to a Baysor cluster.")
        if self.verbose: printwtime(f"In total, {tat} transcript counts are assigned to a segmentation cell, and {tot} are unassigned.")
    
    def assign_baysor_toseg_fromdist(self, ambigN = 10, ambigShare = 0.55, Ntries = 2):
        """ Assign Baysor cells to segmentation from mean transcript distance.
        """
        if self.verbose: printwtime(f"  Clean up Baysor cell neighbors.")
        # First clean neighbors that are too far away
        def clean_neighbors_distance(cell, segcutoff=13, baycutoff=12):
            """ Clean neighbors that are too far away
            """
            if len(cell.segneighbors)>0:
                for index in cell.segneighbors[cell.segneighbors_distance_mean>segcutoff]:
                    cell.drop_segneighbor(index)
            if len(cell.baysorneighbors)>0:
                for index in cell.baysorneighbors[cell.baysorneighbors_distance_mean>segcutoff]:
                    cell.drop_baysorneighbor(index)
        def clean_neighbors_cluster(cell, dobaysor=False):
            """ Clean neighbors that already have a conflicting cluster assignment
            """
            if len(cell.segneighbors)>0:
                clusters = self.cellsseg.loc[cell.segneighbors].cell.cget("cluster")
                for index in cell.segneighbors[np.logical_and(clusters!=cell.cluster, clusters!=0)]:
                    cell.drop_segneighbor(index)
            if dobaysor and len(cell.baysorneighbors)>0:
                for index in cell.baysorneighbors[self.cellsbay.loc[cell.baysorneighbors].cell.cget("cluster")!=cell.cluster]:
                    cell.drop_baysorneighbor(index)
        self.cellsbay[self._cgetbay("segcell")==0].apply(lambda cell: clean_neighbors_distance(cell, segcutoff=13, baycutoff=12))
        self.cellsbay[self._cgetbay("segcell")==0].apply(lambda cell: clean_neighbors_cluster(cell, dobaysor=True))
        # And set which cells can maybe be assigned
        def set_try_segassign():
            """ Unassigned cells that have segneighbors
            """
            self._csetbay("try_assign", np.logical_and(self._cgetbay("segcell")==0,self._cgetbay("segneighbors").apply(len)>0))
        set_try_segassign()

        unassigned = self.cellsbay[self._cgetbay("try_assign")]
        tat = unassigned.cell.cget("transcriptcount").sum()
        if self.verbose: printwtime(f"  Left with {len(unassigned)} Baysor cells with {tat} transcripts that could be assigned to a segneighbor.")
        
        # Do rounds of assignment and cleaning
        for i in range(Ntries):
            if self.verbose: printwtime(f"  Starting round {i} of assignment from distances.")
            self.assign_do_single_tryassign_fromdistance(unassigned)

            self.cellsbay[self._cgetbay("segcell")==0].apply(lambda cell: clean_neighbors_cluster(cell, dobaysor=False))
            set_try_segassign()

            unassigned = self.cellsbay[self._cgetbay("try_assign")]
            tat = unassigned.cell.cget("transcriptcount").sum()
            if self.verbose: printwtime(f"  Left with {len(unassigned)} Baysor cells with {tat} transcripts that could be assigned to a segneighbor.")
        
        self.assign_print_stats()
    
    ##############################
    ### Produce counts for segmentation cells
    ##############################
    
    def collect_assigned_tosegcells(self, reset_first = True):
        """ Collect assigned Baysor cells to segmentation cells.
        """
        if reset_first: self._csetseg("baysorcells", [])
        assigned = self.cellsbay[self._cgetbay("segcell")!=0].cell.cget("segcell")
        for cbay, cseg in assigned.items():
            self.cellsseg.loc[cseg].baysorcells += [cbay]
    
    def assign_counts_tosegcells(self, collect_first = True):
        """ Collect assigned Baysor cells to segmentation cells,
            and assign their counts to them.
        """
        if self.verbose: printwtime(f"Assigning counts to segmentation cells.")
        if collect_first: self.collect_assigned_tosegcells(True)
        
        default = self.cellsbay.iloc[0].counts.copy()
        default[:] = 0
        
        totalcounts = []
        totalcells = 0
        
        for cell in self.cellsseg.iloc:
            if len(cell.baysorcells)==0:
                cell.counts = default.copy()
            else:
                totalcells += 1
                cell.counts = self.cellsbay.loc[cell.baysorcells].cell.cget("counts").sum(axis=0)
                totalcounts += [cell.counts.sum()]
                cell.tpositions = np.concatenate(list(self.cellsbay.loc[cell.baysorcells].cell.cget("tpositions")), axis=0)
        
        if self.verbose: printwtime(f"  Assigned {sum(totalcounts)} transcripts to {totalcells} cells.")
        median = np.round(np.median(totalcounts),1)
        mean = np.round(np.mean(totalcounts),1)
        if self.verbose: printwtime(f"  The mean cell has {mean:.1f} counts, the median is {median:.1f}.")
    
    def make_adata_segcells(self, clusternamedict = {}):
        """ Make adata from segmentation cells.
        """
        counts = self._cgetseg("counts").copy()
        counts.index = counts.index.astype(str)

        obs = pd.DataFrame(self._cgetseg("index"), columns=["MaskIndex"])
        obs["CellName"] = self._cgetseg("name")
        obs["BrainRegion"] = self._cgetseg("brainregion")
        obs["BrainRegionName"] = self._cgetseg("brainregionname")
        obs["ROI"] = self._cgetseg("roi")
        obs["BaysorCluster"] = self._cgetseg("cluster")
        obs[["x","y","z"]] = np.asarray(list(self._cgetseg("position")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            obs[["xT","yT","zT"]] = np.asarray(list(self._cgetseg("tpositions").apply(lambda x: x.mean(axis=0))))
        obs = obs.copy()
        obs.index = obs.index.astype(str)
        var = self.adata_baysor.var.loc[:,self.adata_baysor.var.columns!='Count'].copy()
        var.index = var.index.astype(str)

        adata = anndata.AnnData(counts, obs = obs, var = var )

        adata.obs["TotalGeneCount"] = counts.sum(axis=1)
        adata.obs["MouseGeneCount"] = counts[adata.var.loc[adata.var["Species"]=="Mouse","GeneR"]].sum(axis=1)
        adata.obs["HumanGeneCount"] = counts[adata.var.loc[adata.var["Species"]=="Human","GeneR"]].sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            adata.obs["LogMouseGeneCount"] = np.log(adata.obs["MouseGeneCount"])
            adata.obs["LogHumanGeneCount"] = np.log(adata.obs["HumanGeneCount"])
        adata.obs["MouseGeneShare"] = counts[adata.var.loc[adata.var["Species"]=="Mouse","GeneR"]].sum(axis=1)/counts.sum(axis=1)
        adata.obs["HumanGeneShare"] = counts[adata.var.loc[adata.var["Species"]=="Human","GeneR"]].sum(axis=1)/counts.sum(axis=1)
        
        if len(clusternamedict)>0:
            adata.obs["BaysorClusterCelltype"] = adata.obs["BaysorCluster"].apply(lambda x: clusternamedict[x])
        
        self.adata = adata
        return adata
    
    def make_adata_baysorcells(self, clusternamedict = {}):
        """ Make adata from Baysor cells. 
            Similar to result from assign_counts_from_Baysor, but includes assignment and better position.
        """
        self.adata_baysor.obs[["x","y","z"]] = np.asarray(list(self.cellsbay.cell.cget("position")))
        self.adata_baysor.obs["assigned_to"] = self.cellsbay.cell.cget("segcell")
        self.adata_baysor.obs["BaysorCluster"] = self.adata_baysor.obs["cluster"]
        if len(clusternamedict)>0:
            self.adata_baysor.obs["BaysorClusterCelltype"] = self.adata_baysor.obs["BaysorCluster"].apply(lambda x: clusternamedict[x])

        return self.adata_baysor

##############################
### Util function
##############################

def apply_combine_baysor_output(resultfolder, segloomfile, genemetafile, boundaryfile, idfile, background="", plotpath="", plotwbackpath=""):
    cluster_combine_list = np.load(idfile, allow_pickle=True)["cluster_combine_list"]
    clusternamedict = np.load(idfile, allow_pickle=True)["clusternamedict"].item()
    
    roidata = SegmentedResolveROI(resultfolder, segloomfile, genemetafile)
    
    roidata.cluster_combine(cluster_combine_list)
    roidata.add_baysor_initial()
    roidata.add_direct_connectivity()
    roidata.add_baysor_neighbors(10)
    roidata.add_baysor_segneighbors(boundaryfile, 30)
    roidata.resolve_nonunique_direct(0.82)
    roidata.assign_baysor_toseg_direct(10, 0.6)
    roidata.assign_baysor_toseg_fromdist(10, 0.55, 1)
    roidata.assign_baysor_toseg_fromdist(8, 0.4, 1)
    roidata.assign_counts_tosegcells()
    
    printwtime("Make anndatas")
    roidata.make_adata_segcells(clusternamedict).write_loom(resultfolder+"/segmentation_cells.loom")
    roidata.make_adata_baysorcells(clusternamedict).write_loom(resultfolder+"/baysor_cells_post.loom")
    
    printwtime("Plot assignments")
    plot_final_assignment(roidata, file= plotpath if plotpath else resultfolder+"/cell_assignment.jpeg")
    plt.close()
    if background:
        plot_final_assignment(roidata, background=background, file= plotwbackpath if plotwbackpath else resultfolder+"/cell_assignment_wbackground.jpeg")
        plt.close()