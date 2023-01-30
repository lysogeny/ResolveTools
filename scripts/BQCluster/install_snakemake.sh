#!/bin/bash
module load devel/anaconda/3
module load compiler/gcc/9.3.0
module load devel/python/3.10.0
module load mpi/openmpi/4.1.1
module load system/singularity/3.8.2

pip install git+https://github.com/snakemake/snakemake

export PATH="/home/bq_vwuest/.local/bin:$PATH"