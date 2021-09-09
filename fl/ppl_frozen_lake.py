#!/usr/bin/python3
import argparse
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
_IOD_STRAT_TRAIN_DETERMINISTIC = "frozen_no_repeat"
_IOD_STRAT_TRAIN_STOCHASTIC = "frozen_no_repeat"
_IOD_STRAT_TEST_DETERMINISTIC = "frozen_no_repeat"
_IOD_STRAT_TEST_STOCHASTIC = "frozen_repeat"
_ROLLS_PER_SI_DETERMINISTIC = 1
_ROLLS_PER_SI_TRAIN_STOCHASTIC = 1
_ROLLS_PER_SI_TEST_STOCHASTIC = 30
_SI_SIZES = {4: 11, 8: 53, 12: 114, 16: 203}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--fl-grid-size", type=int, required=True)
    parser.add_argument("--fl-slip-prob", type=float, required=True)
    parser.add_argument("--ppl-num-gens", type=int, required=True)
    parser.add_argument("--ppl-seed", type=int, required=True)
    parser.add_argument("--ppl-pop-size", type=int, required=True)
    parser.add_argument("--ppl-num-elites", type=int, required=True)
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

    train_env = _make_train_env(args)
    test_env = _make_test_env(args)
    is_deterministic = (args.fl_slip_prob == 0)
    si_size = _SI_SIZES[args.fl_grid_size]
    if is_deterministic:
        num_train_rollouts = (si_size * _ROLLS_PER_SI_DETERMINISTIC)
        num_test_rollouts = (si_size * _ROLLS_PER_SI_DETERMINISTIC)
    else:
        assert 0 < args.fl_slip_prob <= 1
        num_train_rollouts = (si_size * _ROLLS_PER_SI_TRAIN_STOCHASTIC)
        num_test_rollouts = (si_size * _ROLLS_PER_SI_TEST_STOCHASTIC)

    ppl_hyperparams = {
        "seed": args.ppl_seed,
        "pop_size": args.ppl_pop_size,
        "num_elites": args.ppl_num_elites,
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

    best_perf_history = {}
    init_pop = ppl.init()
    gen_num = 0
    _calc_pop_stats(gen_num, init_pop, test_env, num_test_rollouts, args,
                    best_perf_history)
    _save_pop(save_path, init_pop, gen_num)
    for gen_num in range(1, args.ppl_num_gens + 1):
        pop = ppl.run_gen()
        _calc_pop_stats(gen_num, pop, test_env, num_test_rollouts, args,
                        best_perf_history)
        _save_pop(save_path, pop, gen_num)

    _save_history(save_path, best_perf_history)
    _save_python_env_info(save_path)
    _save_main_py_script(save_path)


def _setup_save_path(experiment_name):
    save_path = Path(args.experiment_name)
    save_path.mkdir(exist_ok=False)
    return save_path


def _setup_logging(save_path):
    logging.basicConfig(filename=save_path / "experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)


def _make_train_env(args):
    slip_prob = args.fl_slip_prob
    if slip_prob == 0:
        iod_strat = _IOD_STRAT_TRAIN_DETERMINISTIC
    elif 0 < slip_prob < 1:
        iod_strat = _IOD_STRAT_TRAIN_STOCHASTIC
    else:
        assert False
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=slip_prob,
                   iod_strat=iod_strat,
                   seed=_FL_SEED)


def _make_test_env(args):
    slip_prob = args.fl_slip_prob
    if slip_prob == 0:
        iod_strat = _IOD_STRAT_TEST_DETERMINISTIC
    elif 0 < slip_prob < 1:
        iod_strat = _IOD_STRAT_TEST_STOCHASTIC
    else:
        assert False
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=slip_prob,
                   iod_strat=iod_strat,
                   seed=_FL_SEED)


def _calc_pop_stats(gen_num, pop, test_env, num_test_rollouts, args,
                    best_perf_history):
    logging.info(f"gen num {gen_num}")
    fitnesses = [indiv.fitness for indiv in pop]
    min_ = np.min(fitnesses)
    mean = np.mean(fitnesses)
    median = np.median(fitnesses)
    max_ = np.max(fitnesses)
    logging.info(f"min, mean, median, max fitness in pop: {min_}, {mean}, "
                 f"{median}, {max_}")

    non_elites = [indiv for indiv in pop if not indiv.is_elite]
    logging.info(f"num non-elite indivs: {len(non_elites)}")
    total_time_steps_used = sum(
        [indiv.time_steps_used for indiv in non_elites])
    logging.info(f"total time steps used by non-elites: "
                 f"{total_time_steps_used}")

    # find out test perf of max fitness indiv
    best_indiv = sorted(pop, key=lambda indiv: indiv.fitness, reverse=True)[0]
    res = assess_perf(test_env, best_indiv, num_test_rollouts, args.gamma)
    if res.failed:
        logging.info("best failed test perf assessment")
    logging.info(f"test best perf: {res.perf}")
    best_perf_history[gen_num] = res

    # statistics about truncations and failures
    num_truncs = len([
        indiv for indiv in non_elites
        if indiv.perf_assessment_res.time_limit_trunced
    ])
    num_failures = len(
        [indiv for indiv in non_elites if indiv.perf_assessment_res.failed])
    num_non_elites = len(non_elites)
    logging.info(f"Trunc rate = {num_truncs}/{num_non_elites} = "
                 f"{num_truncs/num_non_elites:.4f}")
    logging.info(f"Failure rate = {num_failures}/{num_non_elites} = "
                 f"{num_failures/num_non_elites:.4f}")


def _save_pop(save_path, pop, gen_num):
    with open(save_path / f"pop_gen_{gen_num}.pkl", "wb") as fp:
        pickle.dump(pop, fp)


def _save_history(save_path, best_perf_history):
    with open(save_path / "best_perf_history.pkl", "wb") as fp:
        pickle.dump(best_perf_history, fp)


def _save_python_env_info(save_path):
    result = subprocess.run(["pip3", "freeze"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return_val = result.stdout.decode("utf-8")
    with open(save_path / "python_env_info.txt", "w") as fp:
        fp.write(str(return_val))


def _save_main_py_script(save_path):
    main_file_path = Path(__main__.__file__)
    shutil.copy(main_file_path, save_path)


if __name__ == "__main__":
    set_start_method("spawn")
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")
