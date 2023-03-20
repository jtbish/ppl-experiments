# ppl-experiments

PPL-DL experiments for the paper "Pittsburgh learning classifier systems for explainable reinforcement learning: comparing with XCS" (https://doi.org/10.1145/3512290.3528767)

Most important file is the run script:

```fl/ppl_frozen_lake.py```

this being the script that actually runs PPL-DL on FrozenLake.

Incidental scripts to pass args to this .py file and run on Slurm are:
```fl/ppl_frozen_lake.sh``` and ```fl/run_ppl_frozen_lake.sh```

## Dependencies for run script
rlenvs: https://github.com/jtbish/rlenvs

ppl: https://github.com/jtbish/ppl
