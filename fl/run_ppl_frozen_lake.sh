#!/bin/bash
# variable params
fl_grid_size=4
fl_slip_prob=0.3
fl_iod_strat_base_train="frozen"
fl_iod_strat_base_test="frozen"
# mu val
ppl_rolls_per_si_train_stoca=30

# static / calced params
declare -A ppl_indiv_sizes=( [4]=7 [8]=21 [12]=42 )
ppl_indiv_size="${ppl_indiv_sizes[$fl_grid_size]}"
# pop size = 16x indiv size
declare -A ppl_pop_sizes=( [4]=112 [8]=336 [12]=672 )
ppl_pop_size="${ppl_pop_sizes[$fl_grid_size]}"
ppl_num_gens=50
ppl_tourn_size=3
ppl_p_cross=0.7
ppl_p_cross_swap=0.5
ppl_p_mut=0.01
gamma=0.95

for ppl_seed in {0..0}; do
   sbatch ppl_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$fl_iod_strat_base_train" \
        "$fl_iod_strat_base_test" \
        "$ppl_num_gens" \
        "$ppl_seed" \
        "$ppl_pop_size" \
        "$ppl_indiv_size" \
        "$ppl_tourn_size" \
        "$ppl_p_cross" \
        "$ppl_p_cross_swap" \
        "$ppl_p_mut" \
        "$gamma" \
        "$ppl_rolls_per_si_train_stoca"
done
