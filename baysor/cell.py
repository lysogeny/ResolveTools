import numpy as np
import pandas as pd

##############################
### Baysor Cell
##############################

class BaysorCell:
    """ Segmentated cell from Baysor.
    """
    def __init__(self, series):
        #self.brainregion = series["BrainRegion"]
        #self.brainregionname = series["BrainRegionName"]
        self.name = series["CellName"]
        self.index = series["MaskIndex"]
        self.roi = series["ROI"]
        self.position = np.zeros((0))
        self.cluster = series["cluster"]
        self.transcriptcount = series["n_transcripts"]
        self.segcell = 0
        self.segcellmode = 0
        
        self.tpositions = np.zeros((0))
        self.tclusters = np.zeros((0))
        self.direct_connections = np.zeros((2,0))
        self.baysorneighbors = []
        self.baysorneighbors_distance_mean = np.zeros((0))
        self.baysorneighbors_distance_min = np.zeros((0))
        self.baysorneighbors_distance_max = np.zeros((0))
        self.segneighbors = []
        self.segneighbors_distance_mean = np.zeros((0))
        self.segneighbors_distance_min = np.zeros((0))
        self.segneighbors_distance_max = np.zeros((0))
        self.counts = pd.Series(dtype=int)
        
    def __repr__(self):
        text = "Baysor cell "+self.name+" with index "+str(self.index)+" and cluster "+str(self.cluster)+"."
        text += "\nCurrently assigned to segmentation cell "+str(self.segcell)+"."
        return text
    
    def set_position_from_transcripts(self):
        """ Set centroid from assigned transcripts.
        """
        self.position = self.tpositions.mean(axis=0).astype(float)
    
    def process_direct_connections(self):
        """ Process direct_connections after it was set.
        """
        self.direct_connections_count = self.direct_connections[1][np.argsort(self.direct_connections[1])[::-1]]
        self.direct_connections_index = self.direct_connections[0][np.argsort(self.direct_connections[1])[::-1]]
        self.direct_connections_index_full = self.direct_connections_index.copy()
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
            self.baysorneighbors.remove(self.index)
        self.baysorneighbors = np.asarray(self.baysorneighbors)
    
    def assign_to_segcell(self, index, mode=0):
        """ Assign this cell to a segmentation cell.
        """
        self.segcell = index
        self.segcellmode = mode
        self.try_direct_connection = False
    
    def drop_segneighbor(self, index):
        """ Drop the segmentation cell neighbor at index index.
        """
        self.segneighbors_distance_mean = self.segneighbors_distance_mean[self.segneighbors!=index]
        self.segneighbors_distance_min = self.segneighbors_distance_min[self.segneighbors!=index]
        self.segneighbors_distance_max = self.segneighbors_distance_max[self.segneighbors!=index]
        self.segneighbors = self.segneighbors[self.segneighbors!=index]
    
    def drop_baysorneighbor(self, index):
        """ Drop the Baysor cell neighbor at index index.
        """
        self.baysorneighbors_distance_mean = self.baysorneighbors_distance_mean[self.baysorneighbors!=index]
        self.baysorneighbors_distance_min = self.baysorneighbors_distance_min[self.baysorneighbors!=index]
        self.baysorneighbors_distance_max = self.baysorneighbors_distance_max[self.baysorneighbors!=index]
        self.baysorneighbors = self.baysorneighbors[self.baysorneighbors!=index]

##############################
### Segmentation Cell
##############################

class SegmentationCell:
    """ Segmentated cell from the DAPI image.
    """
    def __init__(self, series):
        self.brainregion = series["BrainRegion"]
        self.brainregionname = series["BrainRegionName"]
        self.name = series["CellName"]
        self.index = series["MaskIndex"]
        self.roi = series["ROI"]
        self.position = np.asarray(series[["x","y","z"]]).astype(float)
        self.cluster = 0
        self.baysorcells = []
        self.counts = pd.Series(dtype=int)
        self.tpositions = np.zeros((0,3))
        
    def __repr__(self):
        text = "Segmentation cell "+self.name+" with index "+str(self.index)+" in region "+self.brainregionname+"."
        text += "\nCurrently assigned to Baysor cluster "+str(self.cluster)
        #text += " with "+str(len(self.baysorcells))+" assigned Baysor cells."
        text += " with assigned Baysor cells "+str(self.baysorcells)+"."
        return text