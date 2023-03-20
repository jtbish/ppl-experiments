#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=3

source ~/virtualenvs/ppl/bin/activate
python3 ppl_frozen_lake.py \
    --experiment-name="$SLURM_JOB_ID" \
    --fl-grid-size="$1" \
    --fl-slip-prob="$2" \
    --fl-iod-strat-base-train="$3" \
    --fl-iod-strat-base-test="$4" \
    --ppl-num-gens="$5" \
    --ppl-seed="$6" \
    --ppl-pop-size="$7" \
    --ppl-indiv-size="$8" \
    --ppl-tourn-size="$9" \
    --ppl-p-cross="${10}" \
    --ppl-p-cross-swap="${11}" \
    --ppl-p-mut="${12}" \
    --gamma="${13}" \
    --ppl-rolls-per-si-train-stoca="${14}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
#mv "${SLURM_JOB_ID}.prof" "${SLURM_JOB_ID}/"
