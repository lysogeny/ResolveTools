#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -p single
#SBATCH -n 10
#SBATCH --mem=50G
#SBATCH -o /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor/logs/%x-%j.olog
#SBATCH -e /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor/logs/%x-%j.elog
#SBATCH --job-name baysor
##SBATCH --mail-type=ALL
##SBATCH --mail-user=valentin.wuest@bioquant.uni-heidelberg.de

. /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/BQCluster/load_modules.sh
cd /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor

mkdir 04_baysor/testN/results_N24_wmesmer_combined
baysor run -c config/resolve.toml -o 04_baysor/testN/results_N24_wmesmer_combined \
		--n-clusters 24 -i 150 --prior-segmentation-confidence 0.98 \
		03_transcripts_combined/T6GBM_transcripts_wmesmer_combined.csv :cell