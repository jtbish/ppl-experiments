import sys
import numpy as np
from collections import namedtuple

Sample = namedtuple("Sample", ["i", "j"])
Subspace = namedtuple("Subspace", ["i_min", "i_max", "j_min", "j_max"])


def main():
    num_buckets = int(sys.argv[1])
    num_samples = int(sys.argv[2])
    assert (num_buckets % 2 == 0 and num_buckets >= 4)
    assert (0 < num_samples <= num_buckets)

    subspaces = _make_subspaces(num_buckets)
    samples = []
    for _ in range(num_samples):
        sample = _take_sample(samples, subspaces)
        samples.append(sample)
    print(samples)

    arr = np.zeros((num_buckets, num_buckets), dtype=np.uint8)
    for sample in samples:
        arr[sample.i][sample.j] = 1
    print(arr.T)


def _make_subspaces(num_buckets):
    half_bucket = (num_buckets // 2)
    hb = half_bucket
    nb = num_buckets
    subspaces = []
    # top left
    subspaces.append(Subspace(i_min=0, i_max=hb-1, j_min=0, j_max=hb-1))
    # top right
    subspaces.append(Subspace(i_min=hb, i_max=nb-1, j_min=0, j_max=hb-1))
    # bottom left
    subspaces.append(Subspace(i_min=0, i_max=hb-1, j_min=hb, j_max=nb-1))
    # bottom right
    subspaces.append(Subspace(i_min=hb, i_max=nb-1, j_min=hb, j_max=nb-1))
    for ss in subspaces:
        assert (ss.i_max - ss.i_min + 1) == (num_buckets // 2)
        assert (ss.j_max - ss.j_min + 1) == (num_buckets // 2)
    print(subspaces)
    return subspaces


def _take_sample(existing_samples, subspaces):
    print(type(subspaces))
    valid = False
    while not valid:
        subspace = _sample_subspace(subspaces)
        i = np.random.randint(low=subspace.i_min, high=subspace.i_max+1)
        j = np.random.randint(low=subspace.j_min, high=subspace.j_max+1)
        valid = _check_if_valid(existing_samples, i, j)
    return Sample(i, j)


def _sample_subspace(subspaces):
    idx = np.random.randint(0, len(subspaces))
    return subspaces[idx]


def _check_if_valid(existing_samples, i, j):
    for existing_sample in existing_samples:
        if i == existing_sample.i or j == existing_sample.j:
            return False
    return True


if __name__ == "__main__":
    main()
