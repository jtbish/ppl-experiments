#!/bin/bash
ppl_num_gens=50
ppl_pop_size=1000
ppl_indiv_size_min=2
ppl_indiv_size_max=10
ppl_inference_strat="sp"
ppl_num_elites=50
ppl_tourn_size=2
ppl_p_mut=0.05
ppl_m_nought=0.1
gamma=1.0
num_rollouts=5

for ppl_seed in {0..4}; do
   sbatch ppl_mountain_car.sh \
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
