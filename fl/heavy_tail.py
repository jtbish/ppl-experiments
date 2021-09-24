import math
import numpy as np


def _sample_heavy_tail_mut_prob(num_alleles):
    n = num_alleles  # num decision vars.
    k = math.floor(n / 2)
    beta = 1.5
    normalisation_const = sum([i**(-1 * beta) for i in range(1, k + 1)])

    dist = {}
    for alpha in range(1, k + 1):
        prob = (1 / normalisation_const) * (alpha**(-1 * beta))
        dist[alpha] = prob
    for (a, p) in dist.items():
        print(a, p)

    chosen_alpha = np.random.choice(a=list(dist.keys()), p=list(dist.values()))
    p_mut = (chosen_alpha / n)
    return p_mut

n = (12*5)
for _ in range(1):
    print(_sample_heavy_tail_mut_prob(num_alleles=n))
