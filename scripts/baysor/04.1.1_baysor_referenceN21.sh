#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -p single
#SBATCH -n 10
#SBATCH -o /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor/logs/%x-%j.olog
#SBATCH -e /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor/logs/%x-%j.elog
#SBATCH --job-name baysor
##SBATCH --mail-type=ALL
##SBATCH --mail-user=valentin.wuest@bioquant.uni-heidelberg.de

. /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/BQCluster/load_modules.sh
cd /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor

mkdir 04_baysor/results_combined_rep
baysor run -c config/resolve.toml -o 04_baysor/results_combined_rep \
		--n-clusters 21 -i 150 --prior-segmentation-confidence 0.98 \
		03_transcripts_combined/T6GBM_transcripts_combined.csv