#!/bin/bash
# variable params
fl_grid_size=4
fl_slip_prob=0
ppl_pop_size_mult=1.5

# static / calced params
# si sizes:
# 4x4 num frozen = 11
# 8x8 num frozen = 53
# 12x12 num frozen = 114
# 16x16 num frozen = 203
declare -A si_sizes=( [4]=11 [8]=53 [12]=114 [16]=203 )
si_size="${si_sizes[$fl_grid_size]}"
declare -A ppl_indiv_sizes=( [4]=10 [8]=20 [12]=30 [16]=40 )
ppl_indiv_size="${ppl_indiv_sizes[$fl_grid_size]}"
# for determinstic (p_slip == 0):
num_train_rollouts=$si_size
num_test_rollouts=$si_size
# for stochastic (p_slip > 0):
samples_per_si_train=1
samples_per_si_test=10
#num_train_rollouts=$(($si_size * $samples_per_si_train))
#num_test_rollouts=$(($si_size * $samples_per_si_test))

ppl_num_gens=100
alleles_per_rule=5
num_search_dims=$(($ppl_indiv_size * $alleles_per_rule))
ppl_pop_size=$(python3 -c "import math; print(math.ceil($ppl_pop_size_mult * $num_search_dims))")
ppl_inference_strat="dl"
# 0: Left, 1: Down, 2: Right, 3: Up, -1: NULL
ppl_default_action=-1
# num elites approx 5% (rounded up) of pop size s.t. (pop_size - num_elites) % 2 == 0
#ppl_num_elites=$(python3 -c "import math; print(2*math.ceil(math.ceil(0.05*$ppl_pop_size)/2))")
ppl_num_elites=0
# tourn size max of (2, x% pop size)
ppl_tourn_percent=0.05
ppl_tourn_size=$(python3 -c "import math; print(max(2, math.ceil($ppl_tourn_percent*$ppl_pop_size)))")
ppl_p_cross=0.8
ppl_p_cross_swap=0.5
ppl_p_mut=0.025
gamma=0.95

for ppl_seed in {0..29}; do
   echo sbatch ppl_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$ppl_num_gens" \
        "$ppl_seed" \
        "$ppl_pop_size" \
        "$ppl_indiv_size" \
        "$ppl_inference_strat" \
        "$ppl_default_action" \
        "$ppl_num_elites" \
        "$ppl_tourn_size" \
        "$ppl_p_cross" \
        "$ppl_p_cross_swap" \
        "$ppl_p_mut" \
        "$num_train_rollouts" \
        "$num_test_rollouts" \
        "$gamma"
done
