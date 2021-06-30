#!/bin/bash
#SBATCH --partition=coursework
#SBATCH --cpus-per-task=2

source ~/virtualenvs/ppl/bin/activate
python3 ppl_frozen_lake.py \
    --experiment-name="$SLURM_JOB_ID" \
    --fl-grid-size="$1" \
    --fl-slip-prob="$2" \
    --fl-tl-mult="$3" \
    --ppl-num-gens="$4" \
    --ppl-seed="$5" \
    --ppl-pop-size="$6" \
    --ppl-indiv-size-min="$7" \
    --ppl-indiv-size-max="$8" \
    --ppl-inference-strat="$9" \
    --ppl-default-action="${10}" \
    --ppl-num-elites="${11}" \
    --ppl-tourn-size="${12}" \
    --ppl-p-cross="${13}" \
    --ppl-p-mut="${14}" \
    --ppl-m-nought="${15}" \
    --num-train-rollouts="${16}" \
    --num-test-rollouts="${17}" \
    --gamma="${18}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
#mv "${SLURM_JOB_ID}.prof" "${SLURM_JOB_ID}/"
