#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1

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
    --ppl-num-elites="$9" \
    --ppl-tourn-size="${10}" \
    --ppl-p-mut="${11}" \
    --ppl-m-nought="${12}" \
    --gamma="${13}" \
    --num-rollouts="${14}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
