#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -p single
#SBATCH -n 10
#SBATCH --mem=50G
#SBATCH -o /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor/logs/%x-%j.olog
#SBATCH -e /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor/logs/%x-%j.elog
#SBATCH --job-name baysor
##SBATCH --mail-type=ALL
##SBATCH --mail-user=valentin.wuest@bioquant.uni-heidelberg.de

N=21
I=150
conf="scale1"
config="config/resolve_${conf}.toml"
name="results_N${N}_wmesmer_noMCh_combined_lpsc_${conf}"
transcripts="T6GBM_transcripts_wmesmer_noMCh_combined"
combkey="${transcripts}_combinekey"

. /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/BQCluster/load_modules.sh
cd /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data/baysor

mkdir -p "04_baysor/${name}"
mkdir -p "04_baysor/${name}/log"
cp "04_baysor/results_N21_wmesmer_noMCh_noshift_combined_scale3/script_make_id.py" "04_baysor/${name}/script_make_id.py"

#baysor run -c $config -o "04_baysor/${name}" --n-clusters $N -i $I --prior-segmentation-confidence 0.98 "03_transcripts_combined/${transcripts}.csv" :cell

singularity exec --bind /mnt/sds-hd/sd17l002/p/resolve_glioblastoma/data:/data library://valentinwust/resolve/mesmer /data/baysor/scripts/final_do_all.sh $name $combkey