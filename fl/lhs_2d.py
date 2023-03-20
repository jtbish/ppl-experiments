import sys
import numpy as np
from collections import namedtuple

Sample = namedtuple("Sample", ["i", "j"])


def main():
    num_buckets = int(sys.argv[1])
    num_samples = int(sys.argv[2])

    assert (0 < num_samples <= num_buckets)

    samples = []
    for _ in range(num_samples):
        sample = _take_sample(samples, num_buckets)
        samples.append(sample)
    print(samples)

    arr = np.zeros((num_buckets, num_buckets), dtype=np.uint8)
    for sample in samples:
        arr[sample.i][sample.j] = 1
    print(arr.T)


def _take_sample(existing_samples, num_buckets):
    valid = False
    while not valid:
        i = np.random.randint(low=0, high=num_buckets)
        j = np.random.randint(low=0, high=num_buckets)
        valid = _check_if_valid(existing_samples, i, j)
    return Sample(i, j)


def _check_if_valid(existing_samples, i, j):
    for existing_sample in existing_samples:
        if i == existing_sample.i or j == existing_sample.j:
            return False
    return True


if __name__ == "__main__":
    main()
