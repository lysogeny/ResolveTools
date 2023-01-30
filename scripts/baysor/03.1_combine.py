import sys
sys.path.insert(0,'/data/')
from ResolveTools.baysor.utils import combine_baysor_transcripts

import os
path = "/data/baysor/01_transcripts/"
files = [path +f for f in os.listdir(path)]
outfile = "/data/baysor/03_transcripts_combined/T6GBM_transcripts_combined.csv"
combine_baysor_transcripts(files, outfile, shift=7000, cellshift=70000)

path = "/data/baysor/02_transcripts_wmesmer/"
files = [path +f for f in os.listdir(path)]
outfile = "/data/baysor/03_transcripts_combined/T6GBM_transcripts_wmesmer_combined.csv"
combine_baysor_transcripts(files, outfile, shift=7000, cellshift=70000)