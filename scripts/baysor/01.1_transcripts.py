import sys
sys.path.insert(0,'/data/')

from resolve_tools.baysor.utils import counts_resolve_to_baysor
from resolve_tools.utils.utils import printwtime
import re
import os

filename = os.path.basename(sys.argv[1])
roikey = re.search(r'R(\d)_W(\d)A(\d)', filename).group(0)

counts_resolve_to_baysor("/data/resolve/04_registration3D/"+filename,
                         "/data/baysor/01_transcripts/"+filename.replace(".txt","_baysor.txt"))

counts_resolve_to_baysor("/data/resolve/04_registration3D/"+filename,
                         "/data/baysor/02_transcripts_wmesmer/"+filename.replace(".txt","_baysor_wmesmer.txt"),
                        segmaskpath="/data/confocal/02_mesmer_nuclei/Confocal_"+roikey+"_DAPI_mesmer_nuclei_post.npz")
