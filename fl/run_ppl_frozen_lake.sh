#!/bin/bash
# variable params
fl_grid_size=4
fl_slip_prob=0

# static / calced params
declare -A ppl_indiv_sizes=( [4]=10 [8]=20 [12]=30 [16]=40 )
ppl_indiv_size="${ppl_indiv_sizes[$fl_grid_size]}"
ppl_num_elites=0
declare -A ppl_pop_sizes=( [4]=100 [8]=200 [12]=300 [16]=400 )
ppl_pop_size="${ppl_pop_sizes[$fl_grid_size]}"

ppl_num_gens=150
#ppl_tourn_percent=0.015
#ppl_tourn_size=$(python3 -c "import math; print(max(2, math.ceil($ppl_tourn_percent * $ppl_pop_size)))")
ppl_tourn_size=2
ppl_p_cross=0.8
ppl_p_cross_swap=0.5
# 4 cond. alleles + 1 action allele
alleles_per_rule=5
num_search_dims=$(($ppl_indiv_size * $alleles_per_rule))
ppl_p_mut=$(bc -l <<< "1 / $num_search_dims")
gamma=0.95

for ppl_seed in {0..29}; do
   sbatch ppl_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$ppl_num_gens" \
        "$ppl_seed" \
        "$ppl_pop_size" \
        "$ppl_num_elites" \
        "$ppl_indiv_size" \
        "$ppl_tourn_size" \
        "$ppl_p_cross" \
        "$ppl_p_cross_swap" \
        "$ppl_p_mut" \
        "$gamma"
done
