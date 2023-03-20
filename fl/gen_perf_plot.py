import glob
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import parse as ps

#matplotlib.use("Agg")

_EXPECTED_NUM_EXP_DIRS = 30
_NUM_GENS = 250
_EXPECTED_HISTORY_LEN = (_NUM_GENS + 1)
_EMP_OPT_PERFS_PKL_FILE = \
    "FrozenLake_emp_opt_perfs_gamma_0.95_iodsb_frozen.pkl"
_STD_ALPHA = 0.25


def main():
    base_dir = sys.argv[1]
    base_dir_tail = Path(base_dir).name
    if "detrm" in base_dir:
        gs = int(ps.parse("gs_{}", base_dir_tail)[0])
        sp = 0
    elif "stoca" in base_dir:
        gs = int(ps.parse("gs_{}_sp_{}", base_dir_tail)[0])
        sp = float(ps.parse("gs_{}_sp_{}", base_dir_tail)[1])
    else:
        assert False
    print(f"Grid size {gs}")
    print(f"Slip prob {sp}")

    # get emp opt perf
    with open(f"./{_EMP_OPT_PERFS_PKL_FILE}", "rb") as fp:
        empirical_opt_perfs = pickle.load(fp)
    emp_opt_perf = empirical_opt_perfs[(gs, sp)].perf

    exp_dirs = glob.glob(f"{base_dir}/65*")
    assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS

    # init aggregate history
    aggr_perf_history = OrderedDict(
        {gen: []
         for gen in range(0, _EXPECTED_HISTORY_LEN)})

    # fill aggregate history
    for exp_dir in exp_dirs:
        with open(f"{exp_dir}/best_indiv_test_perf_history.pkl", "rb") as fp:
            perf_history = pickle.load(fp)
        assert len(perf_history) == _EXPECTED_HISTORY_LEN

        # for history, key is num gens
        for (k, v) in perf_history.items():
            assert k in aggr_perf_history
            # v is perf assessment response
            perf = v.perf
            perf_pcnt = (perf / emp_opt_perf)
            aggr_perf_history[k].append(perf_pcnt)

    # check aggregate history is valid
    for (k, v) in aggr_perf_history.items():
        assert len(v) == _EXPECTED_NUM_EXP_DIRS

    mean_perfs = {k: np.mean(v) for (k, v) in aggr_perf_history.items()}
    std_perfs = {k: np.std(v) for (k, v) in aggr_perf_history.items()}
    last_key = max(mean_perfs.keys())
    last_mean = mean_perfs[last_key]
    last_std = std_perfs[last_key]
    print(f"Perf @ {last_key} gens: {last_mean:.4f} +- {last_std:.4f}")

    plt.figure()
    xs = np.array(list(mean_perfs.keys()))
    ys = np.array(list(mean_perfs.values()))
    err = np.array(list(std_perfs.values()))
    plt.plot(xs, ys)
    plt.fill_between(xs, ys-err, ys+err, alpha=_STD_ALPHA)
    plt.axhline(y=1.0, color="red")
    plt.xlabel("Num generations")
    plt.ylabel("Optimal perf. frac.")
    #plt.show()
    plt.savefig(f"{base_dir}/{base_dir_tail}_perf_plot.png")


if __name__ == "__main__":
    main()
