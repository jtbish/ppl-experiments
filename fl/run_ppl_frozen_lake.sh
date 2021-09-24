#!/bin/bash
# variable params
fl_grid_size=8
fl_slip_prob=0
fl_iod_strat_base="frozen"

# static / calced params
declare -A ppl_indiv_sizes=( [4]=6 [8]=18 [12]=30 )
ppl_indiv_size="${ppl_indiv_sizes[$fl_grid_size]}"
declare -A ppl_pop_sizes=( [4]=90 [8]=270 [12]=450 )
ppl_pop_size="${ppl_pop_sizes[$fl_grid_size]}"
ppl_num_gens=200
ppl_tourn_size=3
ppl_p_cross=0.7
ppl_p_cross_swap=0.5
# 4 cond. alleles + 1 action allele
alleles_per_rule=5
num_search_dims=$(($ppl_indiv_size * $alleles_per_rule))
ppl_p_mut=$(bc -l <<< "2 / $num_search_dims")
gamma=0.95

for ppl_seed in {0..0}; do
   sbatch ppl_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$fl_iod_strat_base" \
        "$ppl_num_gens" \
        "$ppl_seed" \
        "$ppl_pop_size" \
        "$ppl_indiv_size" \
        "$ppl_tourn_size" \
        "$ppl_p_cross" \
        "$ppl_p_cross_swap" \
        "$ppl_p_mut" \
        "$gamma"
done
