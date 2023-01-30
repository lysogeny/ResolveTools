#!/bin/bash
#SBATCH -t 0-02:00:00
#SBATCH -p single
#SBATCH -n 10
#SBATCH --mem=50G
#SBATCH -o /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/confocal/logs/%x-%j.olog
#SBATCH -e /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/confocal/logs/%x-%j.elog
#SBATCH --job-name clip_copy
##SBATCH --mail-type=ALL
##SBATCH --mail-user=valentin.wuest@bioquant.uni-heidelberg.de

. /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/BQCluster/load_modules.sh
##. /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/BQCluster/run_singularity_mesmer.sh

singularity exec --bind /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data:/data library://valentinwust/resolve/mesmer python /data/confocal/01.2.1_clip_copy.py "$@"

##cd /data/confocal
##python 01.2.1_clip_copy.py "$@"