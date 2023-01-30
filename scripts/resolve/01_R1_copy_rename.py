import numpy as np
import os

files = os.listdir("../resolve_raw/Run1/")
files = list(filter(lambda x: "Channel0_R1" in x,files))
files = [file.replace("Panorama_T6GBM_","").replace("_Channel0_R1_.tiff","") for file in files]

files = list(filter(lambda x: x not in ["W4A1", "W4A2"], files)) # Failed runs

copy_from = ["../resolve_raw/Run1/Panorama_T6GBM_"+file+"_results.txt" for file in files]
copy_to = ["01_raw/T6GBM_R1_"+file+"_transcripts.txt" for file in files]
for fr, to in zip(copy_from, copy_to):
    os.system("cp "+fr+" "+to)

copy_from = ["../resolve_raw/Run1/Panorama_T6GBM_"+file+"_Channel3_R8_.tiff" for file in files]
copy_to = ["01_raw/T6GBM_R1_"+file+"_DAPI.tiff" for file in files]
for fr, to in zip(copy_from, copy_to):
    os.system("cp "+fr+" "+to)