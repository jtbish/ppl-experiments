#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4

source ~/virtualenvs/ppl/bin/activate
python3 ppl_mountain_car.py \
    --experiment-name="$SLURM_JOB_ID" \
    --ppl-num-gens="$1" \
    --ppl-seed="$2" \
    --ppl-pop-size="$3" \
    --ppl-indiv-size-min="$4" \
    --ppl-indiv-size-max="$5" \
    --ppl-inference-strat="$6" \
    --ppl-num-elites="$7" \
    --ppl-tourn-size="$8" \
    --ppl-p-mut="$9" \
    --ppl-m-nought="${10}" \
    --gamma="${11}" \
    --num-rollouts="${12}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
