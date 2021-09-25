#!/bin/bash
# variable params
fl_grid_size=4
fl_slip_prob=0
fl_iod_strat_base="frozen"

# static / calced params
declare -A ppl_indiv_sizes=( [4]=6 [8]=18 [12]=30 )
ppl_indiv_size="${ppl_indiv_sizes[$fl_grid_size]}"
declare -A ppl_pop_sizes=( [4]=90 [8]=270 [12]=450 )
ppl_pop_size="${ppl_pop_sizes[$fl_grid_size]}"
ppl_num_gens=200
ppl_tourn_size=2
ppl_p_cross=0.7
ppl_p_cross_swap=0.5
ppl_p_mut=0.01
gamma=0.95

for ppl_seed in {0..29}; do
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
