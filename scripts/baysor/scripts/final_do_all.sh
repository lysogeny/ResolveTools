#!/bin/bash
python "/data/baysor/04_baysor/$1/script_make_id.py" "$@" 2>&1 | tee "/data/baysor/04_baysor/$1/log/00_make_id.o"
python /data/baysor/scripts/final_01_process_simple.py "$@" 2>&1 | tee "/data/baysor/04_baysor/$1/log/01_simple_assign.o"
python /data/baysor/scripts/final_02_process_custom.py "$@" 2>&1 | tee "/data/baysor/04_baysor/$1/log/02_custom_assign.o"
python /data/baysor/scripts/final_03_add_region.py "$@" 2>&1 | tee "/data/baysor/04_baysor/$1/log/03_add_region.o"
python /data/baysor/scripts/final_04_add_stain.py "$@" 2>&1 | tee "/data/baysor/04_baysor/$1/log/04_add_stain.o"
python /data/baysor/scripts/final_05_combine_final.py "$@" 2>&1 | tee "/data/baysor/04_baysor/$1/log/05_combine_final.o"
python /data/baysor/scripts/final_06_add_QC.py "$@" 2>&1 | tee "/data/baysor/04_baysor/$1/log/06_add_QC.o"
python /data/baysor/scripts/final_07_assignment_plots.py "$@" 2>&1 | tee "/data/baysor/04_baysor/$1/log/07_assignment_plots.o"
