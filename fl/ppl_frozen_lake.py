#!/usr/bin/python3
import argparse
import glob
import logging
import os
import pickle
import shutil
import subprocess
import time
from multiprocessing import set_start_method
from pathlib import Path

import __main__
import numpy as np
from ppl.encoding import IntegerUnorderedBoundEncoding
from ppl.ppl import PPL
from rlenvs.environment import assess_perf
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
_FL_SEED = 0
_ROLLS_PER_SI_DETERMINISTIC = 1
_ROLLS_PER_SI_TRAIN_STOCHASTIC = 1
_ROLLS_PER_SI_TEST_STOCHASTIC = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--fl-grid-size", type=int, required=True)
    parser.add_argument("--fl-slip-prob", type=float, required=True)
    parser.add_argument("--fl-iod-strat-base", required=True)
    parser.add_argument("--ppl-num-gens", type=int, required=True)
    parser.add_argument("--ppl-seed", type=int, required=True)
    parser.add_argument("--ppl-pop-size", type=int, required=True)
    parser.add_argument("--ppl-indiv-size", type=int, required=True)
    parser.add_argument("--ppl-tourn-size", type=int, required=True)
    parser.add_argument("--ppl-p-cross", type=float, required=True)
    parser.add_argument("--ppl-p-cross-swap", type=float, required=True)
    parser.add_argument("--ppl-p-mut", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    if args.fl_slip_prob == 0:
        is_deterministic = True
    elif 0 < args.fl_slip_prob < 1:
        is_deterministic = False
    else:
        assert False

    iod_strat_base = args.fl_iod_strat_base
    if is_deterministic:
        iod_strat_train = iod_strat_base + "_no_repeat"
        iod_strat_test = iod_strat_train
    else:
        iod_strat_train = iod_strat_base + "_no_repeat"
        iod_strat_test = iod_strat_base + "_repeat"
    train_env = _make_env(args, iod_strat_train)
    test_env = _make_env(args, iod_strat_test)

    assert train_env.si_size == test_env.si_size
    si_size = train_env.si_size

    if is_deterministic:
        num_train_rollouts = (si_size * _ROLLS_PER_SI_DETERMINISTIC)
        num_test_rollouts = num_train_rollouts
    else:
        num_train_rollouts = (si_size * _ROLLS_PER_SI_TRAIN_STOCHASTIC)
        num_test_rollouts = (si_size * _ROLLS_PER_SI_TEST_STOCHASTIC)

    ppl_hyperparams = {
        "seed": args.ppl_seed,
        "pop_size": args.ppl_pop_size,
        "indiv_size": args.ppl_indiv_size,
        "tourn_size": args.ppl_tourn_size,
        "p_cross": args.ppl_p_cross,
        "p_cross_swap": args.ppl_p_cross_swap,
        "p_mut": args.ppl_p_mut,
        "num_rollouts": num_train_rollouts,
        "gamma": args.gamma
    }
    logging.info(ppl_hyperparams)
    encoding = IntegerUnorderedBoundEncoding(train_env.obs_space)
    ppl = PPL(train_env, encoding, hyperparams_dict=ppl_hyperparams)

    best_test_perf_history = {}
    init_pop = ppl.init()
    gen_num = 0
    _calc_pop_stats(gen_num, init_pop, test_env, num_test_rollouts, args,
                    best_test_perf_history)
    _save_pop(save_path, init_pop, gen_num)
    num_gens = args.ppl_num_gens
    for gen_num in range(1, num_gens + 1):
        pop = ppl.run_gen()
        _calc_pop_stats(gen_num, pop, test_env, num_test_rollouts, args,
                        best_test_perf_history)
        _save_pop(save_path, pop, gen_num)

    _save_history(save_path, best_test_perf_history)
    _save_python_env_info(save_path)
    _save_main_py_script(save_path)
    _compress_pop_pkl_files(save_path, num_gens)
    _delete_uncompressed_pop_pkl_files(save_path)


def _setup_save_path(experiment_name):
    save_path = Path(args.experiment_name)
    save_path.mkdir(exist_ok=False)
    return save_path


def _setup_logging(save_path):
    logging.basicConfig(filename=save_path / "experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)


def _make_env(args, iod_strat):
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=args.fl_slip_prob,
                   iod_strat=iod_strat,
                   seed=_FL_SEED)


def _calc_pop_stats(gen_num, pop, test_env, num_test_rollouts, args,
                    best_test_perf_history):
    logging.info(f"gen num {gen_num}")
    fitnesses = [indiv.fitness for indiv in pop]
    min_ = np.min(fitnesses)
    mean = np.mean(fitnesses)
    median = np.median(fitnesses)
    max_ = np.max(fitnesses)
    logging.info(f"min, mean, median, max fitness in pop: {min_}, {mean}, "
                 f"{median}, {max_}")

    # find out test perf of max fitness indiv
    best_indiv = sorted(pop, key=lambda indiv: indiv.fitness, reverse=True)[0]
    res = assess_perf(test_env, best_indiv, num_test_rollouts, args.gamma)
    logging.info(f"best test perf assess res: {res}")
    best_test_perf_history[gen_num] = res

    # statistics about truncations and failures
    num_truncs = len([
        indiv for indiv in pop if indiv.perf_assessment_res.time_limit_trunced
    ])
    num_failures = len(
        [indiv for indiv in pop if indiv.perf_assessment_res.failed])
    pop_size = len(pop)
    logging.info(f"Trunc rate = {num_truncs}/{pop_size} = "
                 f"{num_truncs/pop_size:.4f}")
    logging.info(f"Failure rate = {num_failures}/{pop_size} = "
                 f"{num_failures/pop_size:.4f}")


def _save_pop(save_path, pop, gen_num):
    with open(save_path / f"pop_gen_{gen_num}.pkl", "wb") as fp:
        pickle.dump(pop, fp)


def _save_history(save_path, best_test_perf_history):
    with open(save_path / "best_test_perf_history.pkl", "wb") as fp:
        pickle.dump(best_test_perf_history, fp)


def _save_python_env_info(save_path):
    result = subprocess.run(["pip3", "freeze", "-vvv"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return_val = result.stdout.decode("utf-8")
    with open(save_path / "python_env_info.txt", "w") as fp:
        fp.write(str(return_val))


def _save_main_py_script(save_path):
    main_file_path = Path(__main__.__file__)
    shutil.copy(main_file_path, save_path)


def _compress_pop_pkl_files(save_path, num_gens):
    pop_pkl_files = glob.glob(f"{save_path}/pop*.pkl")
    assert len(pop_pkl_files) == (num_gens + 1)
    # use max xz compression: seems to give about 80% reduced size
    os.environ["XZ_OPT"] = "-9e"
    subprocess.run(["tar", "-cJf", f"{save_path}/pops.tar.xz"] + pop_pkl_files,
                   check=True)


def _delete_uncompressed_pop_pkl_files(save_path):
    pop_pkl_files = glob.glob(f"{save_path}/pop*.pkl")
    for file_ in pop_pkl_files:
        os.remove(file_)


if __name__ == "__main__":
    set_start_method("spawn")
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")
