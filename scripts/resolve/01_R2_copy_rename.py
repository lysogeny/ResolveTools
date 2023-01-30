import numpy as np
import os
import string
def new_name(name):
    new = "W"
    if name[1]=="1": new += str(string.ascii_uppercase.index(name[0]))
    else: new += str(string.ascii_uppercase.index(name[0])+4)
    new += "A"+str(name[3])
    return new

files = os.listdir("../resolve_raw/Run2/")
files = list(filter(lambda x: "DAPI" in x,files))
files = [file.replace("T6GBM_redo_","").replace("_DAPI.tiff","") for file in files]
names = [new_name(name) for name in files]

copy_from = ["../resolve_raw/Run2/T6GBM_redo_"+file+"_results.txt" for file in files]
copy_to = ["01_raw/T6GBM_R2_"+file+"_transcripts.txt" for file in names]
for fr, to in zip(copy_from, copy_to):
    os.system("cp "+fr+" "+to)

copy_from = ["../resolve_raw/Run2/T6GBM_redo_"+file+"_DAPI.tiff" for file in files]
copy_to = ["01_raw/T6GBM_R2_"+file+"_DAPI.tiff" for file in names]
for fr, to in zip(copy_from, copy_to):
    os.system("cp "+fr+" "+to)