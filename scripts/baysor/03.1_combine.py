import os
import sys
sys.path.insert(0,'/data/')
from resolve_tools.baysor.utils import combine_baysor_transcripts



path = "/data/baysor/01_transcripts/"
files = [path +f for f in os.listdir(path)]
outfile = "/data/baysor/03_transcripts_combined/T6GBM_transcripts_combined.csv"
combine_baysor_transcripts(files, outfile, shift=7000, cellshift=70000)

path = "/data/baysor/01_transcripts/"
files = [path +f for f in os.listdir(path)]
outfile = "/data/baysor/03_transcripts_combined/T6GBM_transcripts_noMCh_combined.csv"
combine_baysor_transcripts(files, outfile, shift=7000, cellshift=70000, dropgenes=["MCHERRY"])



path = "/data/baysor/02_transcripts_wmesmer/"
files = [path +f for f in os.listdir(path)]
outfile = "/data/baysor/03_transcripts_combined/T6GBM_transcripts_wmesmer_combined.csv"
combine_baysor_transcripts(files, outfile, shift=7000, cellshift=70000)

path = "/data/baysor/02_transcripts_wmesmer/"
files = [path +f for f in os.listdir(path)]
outfile = "/data/baysor/03_transcripts_combined/T6GBM_transcripts_wmesmer_noMCh_combined.csv"
combine_baysor_transcripts(files, outfile, shift=7000, cellshift=70000, dropgenes=["MCHERRY"])
