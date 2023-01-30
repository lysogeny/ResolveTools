import sys
sys.path.insert(0,'/data/')
from ResolveTools.baysor.utils import save_clusterids, load_multiple_ROIs_transcripts, cluster_crosstab

resultfolder = "/data/baysor/04_baysor/reference_N21"
genemetafile = "/data/metadata/gbm_resolve_genes.csv"

transcripts, transcripts_wnoise = load_multiple_ROIs_transcripts(resultfolder, genemetafile, True)

cluster_combine_list = [[3,8],[3,11],[3,16],[3,19],[9,13],[9,15],[9,20],[2,7],[2,14],[2,17],[2,18]]
clusternamedict = {   0:"unknown",
                      1:"OD",
                      2:"Human Q/A",
                      9:"Neuron",
                      5:"Human Q",
                      4:"Macrophages",
                      10:"Endothelial",
                      6:"Microglia",
                      3:"Astrocyte",
                      12:"Human D",
                      21:"Human A"}
humannamecolordict = {'Astrocyte':"black",
                      'Endothelial':"black",
                      'Human A':"red",
                      'Human D':"blue",
                      'Human Q':"green",
                      'Human Q/A':"orange",
                      'Macrophages':"black",
                      'Microglia':"black",
                      'Neuron':"black",
                      'OD':"black",
                      'unknown':"gray"}
mousenamecolordict = {'Astrocyte':"blue",
                      'Endothelial':"red",
                      'Human A':"black",
                      'Human D':"black",
                      'Human Q':"black",
                      'Human Q/A':"black",
                      'Macrophages':"orange",
                      'Microglia':"yellow",
                      'Neuron':"green",
                      'OD':"purple",
                      'unknown':"gray"}
cross = cluster_crosstab(transcripts, wtotal=False)
save_clusterids(resultfolder, cluster_combine_list, clusternamedict, cross, humannamecolordict, mousenamecolordict)