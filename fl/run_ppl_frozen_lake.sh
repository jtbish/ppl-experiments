#!/bin/bash
fl_grid_size=12
fl_slip_prob=0.0
ppl_num_gens=50
ppl_pop_size=200
ppl_indiv_size_min=4
ppl_indiv_size_max=30
ppl_inference_strat="dl"
# 0: Left, 1: Down, 2: Right, 3: Up, -1: NULL
ppl_default_action=-1
ppl_num_elites=20
ppl_tourn_size=5
ppl_p_cross=0.8
ppl_p_mut=0.05
ppl_m_nought=$(($fl_grid_size / 4))


# 4x4: 11, 8x8: 53, 12x12: 114, 16x16: 203
fl_size_si=114

num_train_rollouts=$fl_size_si
num_test_rollouts=$fl_size_si

c_eta=20
#num_train_rollouts=$(python3 -c "import math; print(math.ceil($c_eta * $fl_size_si * $fl_slip_prob))")
#tests_per_si=100
#num_test_rollouts=$(($fl_size_si * $tests_per_si))

gamma=0.95

for ppl_seed in {0..0}; do
   echo sbatch ppl_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$ppl_num_gens" \
        "$ppl_seed" \
        "$ppl_pop_size" \
        "$ppl_indiv_size_min" \
        "$ppl_indiv_size_max" \
        "$ppl_inference_strat" \
        "$ppl_default_action" \
        "$ppl_num_elites" \
        "$ppl_tourn_size" \
        "$ppl_p_cross" \
        "$ppl_p_mut" \
        "$ppl_m_nought" \
        "$num_train_rollouts" \
        "$num_test_rollouts" \
        "$gamma"
done
