#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4

source ~/virtualenvs/ppl/bin/activate
python3 ppl_frozen_lake.py \
    --experiment-name="$SLURM_JOB_ID" \
    --fl-grid-size="$1" \
    --fl-slip-prob="$2" \
    --fl-iod-strat-base="$3" \
    --ppl-num-gens="$4" \
    --ppl-seed="$5" \
    --ppl-pop-size="$6" \
    --ppl-indiv-size="$7" \
    --ppl-tourn-size="$8" \
    --ppl-p-cross="$9" \
    --ppl-p-cross-swap="${10}" \
    --ppl-p-mut="${11}" \
    --gamma="${12}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
