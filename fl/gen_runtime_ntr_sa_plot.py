import math
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#matplotlib.use("Agg")

_EXPECTED_NUM_RUNTIMES = 30
_MU_VALS = (1, 2, 4, 8, 12, 16, 30)
_BASE_DIR = "./frozen/num_train_rolls_sa"


def main():
    mu_runtime_map = OrderedDict({mu: [] for mu in _MU_VALS})

    # fill runtime map
    for mu in _MU_VALS:
        runtimes_txt = f"{_BASE_DIR}/{mu}x_si/runtimes_cut.txt"
        with open(runtimes_txt, "r") as fp:
            for line in fp:
                val = _round_nearest_int(float(line))
                mu_runtime_map[mu].append(val)

    # check runtime map valid
    for v in mu_runtime_map.values():
        assert len(v) == _EXPECTED_NUM_RUNTIMES

    print(mu_runtime_map)
    mean_runtimes_minutes = {
        k: np.mean(v) / 60
        for (k, v) in mu_runtime_map.items()
    }

    plt.figure()
    xs = mean_runtimes_minutes.keys()
    ys = mean_runtimes_minutes.values()
    plt.plot(xs, ys)
    plt.xscale("log", base=2)
    plt.xlabel("$\mu$")
    plt.ylabel("Runtime (minutes)")
    plt.show()


def _round_nearest_int(float_):
    return math.floor(float_ + 0.5)


if __name__ == "__main__":
    main()
