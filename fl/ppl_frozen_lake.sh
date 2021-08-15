#!/bin/bash
#SBATCH --partition=coursework
#SBATCH --cpus-per-task=4

source ~/virtualenvs/ppl/bin/activate
python3 ppl_frozen_lake.py \
    --experiment-name="$SLURM_JOB_ID" \
    --fl-grid-size="$1" \
    --fl-slip-prob="$2" \
    --ppl-num-gens="$3" \
    --ppl-seed="$4" \
    --ppl-pop-size="$5" \
    --ppl-indiv-size="$6" \
    --ppl-inference-strat="$7" \
    --ppl-default-action="$8" \
    --ppl-num-elites="$9" \
    --ppl-tourn-size="${10}" \
    --ppl-p-cross="${11}" \
    --ppl-p-cross-swap="${12}" \
    --ppl-p-mut="${13}" \
    --num-train-rollouts="${14}" \
    --num-test-rollouts="${15}" \
    --gamma="${16}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
#mv "${SLURM_JOB_ID}.prof" "${SLURM_JOB_ID}/"
