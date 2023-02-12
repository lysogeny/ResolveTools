#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -p single
#SBATCH -n 5
#SBATCH --mem=50G
#SBATCH -o /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor/logs/%x-%j.olog
#SBATCH -e /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor/logs/%x-%j.elog
#SBATCH --job-name final_processing
##SBATCH --mail-type=ALL
##SBATCH --mail-user=valentin.wuest@bioquant.uni-heidelberg.de

. /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/BQCluster/load_modules.sh

singularity exec --bind /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data:/data library://valentinwust/resolve/mesmer \
                                               /data/baysor/scripts/final_do_all.sh "$@"

