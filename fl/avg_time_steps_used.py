import glob
import sys

import numpy as np


def main():
    base_dir = sys.argv[1]
    num_gens = int(sys.argv[2])
    experiment_dirs = glob.glob(f"{base_dir}/63*")
    num_experiments = len(experiment_dirs)
    #assert num_experiments == 30

    # rows are experiment dirs, cols are gens (+1 on gens b.c. of gen 0)
    time_steps_mat = np.full((num_experiments, num_gens + 1), np.nan)
    for (row_idx, experiment_dir) in enumerate(experiment_dirs):
        with open(f"{experiment_dir}/time_steps_used.txt", "r") as fp:
            time_steps_used = fp.readlines()
        time_steps_used = np.array([int(e) for e in time_steps_used])
        assert len(time_steps_used) == (num_gens + 1)
        time_steps_mat[row_idx] = time_steps_used
    avg_time_steps_arr = np.ceil(np.mean(time_steps_mat, axis=0)).astype("int")
    assert len(avg_time_steps_arr) == (num_gens + 1)
    np.save(f"{base_dir}/time_steps_mat.npy", time_steps_mat)
    np.save(f"{base_dir}/avg_time_steps_arr.npy", avg_time_steps_arr)
    np.savetxt(f"{base_dir}/avg_time_steps.txt",
               avg_time_steps_arr,
               fmt="%d")


if __name__ == "__main__":
    main()
