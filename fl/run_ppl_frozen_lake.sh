#!/bin/bash
fl_grid_size=4
fl_slip_prob=0.0
ppl_num_gens=50
ppl_pop_size=100
ppl_indiv_size_min=2
ppl_indiv_size_max=10
ppl_inference_strat="dl"
ppl_num_elites=4
ppl_tourn_size=2
ppl_p_mut=0.05
ppl_m_nought=2
gamma=0.95
num_rollouts=1

for ppl_seed in {0..0}; do
   sbatch ppl_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$ppl_num_gens" \
        "$ppl_seed" \
        "$ppl_pop_size" \
        "$ppl_indiv_size_min" \
        "$ppl_indiv_size_max" \
        "$ppl_inference_strat" \
        "$ppl_num_elites" \
        "$ppl_tourn_size" \
        "$ppl_p_mut" \
        "$ppl_m_nought" \
        "$gamma" \
        "$num_rollouts"
done
