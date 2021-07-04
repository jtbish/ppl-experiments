#!/bin/bash
# variable params
fl_grid_size=16
# si sizes:
# 4x4 num frozen = 11
# 8x8 num frozen = 53
# 12x12 num frozen = 114
# 16x16 num frozen = 203
si_size=203
fl_slip_prob=0
ppl_indiv_size=32
ppl_pop_size_mult=10
# for determinstic (p_slip == 0):
num_train_rollouts=$si_size
num_test_rollouts=$si_size
# for stochastic (p_slip > 0):
c_rho=2
tests_per_si=100
#num_train_rollouts=$(python3 -c "import math; print($si_size + math.ceil($c_rho * $si_size * $fl_slip_prob))")
#num_test_rollouts=$(($si_size * $tests_per_si))

# static or calced params
fl_tl_mult=1.5  # time limit mult
ppl_num_gens=100
num_search_dims=$(($ppl_indiv_size * 5))  # 5 alleles for each rule in FL
ppl_pop_size=$(($ppl_pop_size_mult * $num_search_dims))  # N times num search dims
ppl_inference_strat="dl"
# 0: Left, 1: Down, 2: Right, 3: Up, -1: NULL
ppl_default_action=-1
# num elites approx 5% (rounded up) of pop size s.t. (pop_size - num_elites) % 2 == 0
ppl_num_elites=$(python3 -c "import math; print(2*math.ceil(math.ceil(0.05*$ppl_pop_size)/2))")
# tourn size max of (2, 2.5% pop size)
ppl_tourn_size=$(python3 -c "import math; print(max(2, math.ceil(0.025*$ppl_pop_size)))")
ppl_p_cross=0.8
ppl_p_mut=0.05
ppl_m_nought=$(($fl_grid_size / 4))
gamma=0.95

for ppl_seed in {0..29}; do
   echo sbatch ppl_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$fl_tl_mult" \
        "$ppl_num_gens" \
        "$ppl_seed" \
        "$ppl_pop_size" \
        "$ppl_indiv_size" \
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
