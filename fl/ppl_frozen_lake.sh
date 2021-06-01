#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4

source ~/virtualenvs/ppl/bin/activate
python3 ppl_frozen_lake.py \
    --experiment-name="$SLURM_JOB_ID" \
    --fl-grid-size="$1" \
    --fl-slip-prob="$2" \
    --ppl-num-gens="$3" \
    --ppl-seed="$4" \
    --ppl-pop-size="$5" \
    --ppl-indiv-size-min="$6" \
    --ppl-indiv-size-max="$7" \
    --ppl-inference-strat="$8" \
    --ppl-default-action="$9" \
    --ppl-num-elites="${10}" \
    --ppl-tourn-size="${11}" \
    --ppl-p-cross="${12}" \
    --ppl-p-mut="${13}" \
    --ppl-m-nought="${14}" \
    --num-train-rollouts="${15}" \
    --num-test-rollouts="${16}" \
    --gamma="${17}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
