python 04_baysor/results_N21_wmesmer_combined/script_make_id.py
python 04.3.2_process_simple.py 2>&1 | tee /data/baysor/04_baysor/results_N21_wmesmer_combined/simple_assign.o
python 04.3.3_process_custom.py 2>&1 | tee /data/baysor/04_baysor/results_N21_wmesmer_combined/custom_assign.o
python 04.3.5_add_region.py 2>&1 | tee /data/baysor/04_baysor/results_N21_wmesmer_combined/add_region.o
python 04.3.6_combine_final.py 2>&1 | tee /data/baysor/04_baysor/results_N21_wmesmer_combined/combine_final.o