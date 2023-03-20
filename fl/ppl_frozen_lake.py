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
_ROLLS_PER_SI_TEST_STOCHASTIC = 30
_USE_INDIV_POLICY_CACHE = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--fl-grid-size", type=int, required=True)
    parser.add_argument("--fl-slip-prob", type=float, required=True)
    parser.add_argument("--fl-iod-strat-base-train", required=True)
    parser.add_argument("--fl-iod-strat-base-test", required=True)
    parser.add_argument("--ppl-num-gens", type=int, required=True)
    parser.add_argument("--ppl-seed", type=int, required=True)
    parser.add_argument("--ppl-pop-size", type=int, required=True)
    parser.add_argument("--ppl-indiv-size", type=int, required=True)
    parser.add_argument("--ppl-tourn-size", type=int, required=True)
    parser.add_argument("--ppl-p-cross", type=float, required=True)
    parser.add_argument("--ppl-p-cross-swap", type=float, required=True)
    parser.add_argument("--ppl-p-mut", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--ppl-rolls-per-si-train-stoca",
                        type=int,
                        required=True)
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

    if is_deterministic:
        iod_strat_train = args.fl_iod_strat_base_train + "_no_repeat"
        iod_strat_test = args.fl_iod_strat_base_test + "_no_repeat"
    else:
        iod_strat_train = args.fl_iod_strat_base_train + "_repeat"
        iod_strat_test = args.fl_iod_strat_base_test + "_repeat"
    train_env = _make_env(args, iod_strat_train)
    test_env = _make_env(args, iod_strat_test)

    train_si_size = train_env.si_size
    test_si_size = test_env.si_size
    logging.info(f"Training on iod strat: {iod_strat_train}")
    logging.info(f"Testing on iod strat: {iod_strat_test}")
    logging.info(f"Train si size = {train_si_size}")
    logging.info(f"Test si size = {test_si_size}")

    if is_deterministic:
        num_train_rollouts = (train_si_size * _ROLLS_PER_SI_DETERMINISTIC)
        logging.info(f"Using {train_si_size} * {_ROLLS_PER_SI_DETERMINISTIC} "
                     f"= {num_train_rollouts} rollouts for training")
        num_test_rollouts = (test_si_size * _ROLLS_PER_SI_DETERMINISTIC)
        logging.info(f"Using {test_si_size} * {_ROLLS_PER_SI_DETERMINISTIC} "
                     f"= {num_test_rollouts} rollouts for testing")
    else:
        assert args.ppl_rolls_per_si_train_stoca >= 1
        num_train_rollouts = \
            (train_si_size * args.ppl_rolls_per_si_train_stoca)
        logging.info(f"Using {train_si_size} * "
                     f"{args.ppl_rolls_per_si_train_stoca} = "
                     f"{num_train_rollouts} rollouts for training")
        num_test_rollouts = (test_si_size * _ROLLS_PER_SI_TEST_STOCHASTIC)
        logging.info(f"Using {test_si_size} * "
                     f"{_ROLLS_PER_SI_TEST_STOCHASTIC} = "
                     f"{num_test_rollouts} rollouts for testing")

    ppl_hyperparams = {
        "seed": args.ppl_seed,
        "pop_size": args.ppl_pop_size,
        "indiv_size": args.ppl_indiv_size,
        "tourn_size": args.ppl_tourn_size,
        "p_cross": args.ppl_p_cross,
        "p_cross_swap": args.ppl_p_cross_swap,
        "p_mut": args.ppl_p_mut,
        "num_rollouts": num_train_rollouts,
        "gamma": args.gamma,
        "use_indiv_policy_cache": _USE_INDIV_POLICY_CACHE
    }
    logging.info(ppl_hyperparams)
    encoding = IntegerUnorderedBoundEncoding(train_env.obs_space)
    ppl = PPL(train_env, encoding, hyperparams_dict=ppl_hyperparams)

    best_indiv_history = {}
    best_indiv_test_perf_history = {}
    init_pop = ppl.init()
    gen_num = 0
    _calc_pop_stats(gen_num, init_pop, test_env, num_test_rollouts, args,
                    best_indiv_history, best_indiv_test_perf_history)
    _save_ppl(save_path, ppl, gen_num)
    num_gens = args.ppl_num_gens
    for gen_num in range(1, num_gens + 1):
        pop = ppl.run_gen()
        _calc_pop_stats(gen_num, pop, test_env, num_test_rollouts, args,
                        best_indiv_history, best_indiv_test_perf_history)
        _save_ppl(save_path, ppl, gen_num)

    _save_histories(save_path, best_indiv_history,
                    best_indiv_test_perf_history)
    _save_main_py_script(save_path)
#    _compress_ppl_pkl_files(save_path, num_gens)
#    _delete_uncompressed_ppl_pkl_files(save_path)


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
                    best_indiv_history, best_indiv_test_perf_history):
    logging.info(f"gen num {gen_num}")
    fitnesses = [indiv.fitness for indiv in pop]
    min_ = np.min(fitnesses)
    mean = np.mean(fitnesses)
    median = np.median(fitnesses)
    max_ = np.max(fitnesses)
    logging.info(f"min, mean, median, max fitness in pop: {min_}, {mean}, "
                 f"{median}, {max_}")

    # find best indiv and its test perf
    best_indiv = sorted(pop, key=lambda indiv: indiv.fitness, reverse=True)[0]
    res = assess_perf(test_env, best_indiv, num_test_rollouts, args.gamma)
    logging.info(f"best test perf assess res: {res}")
    best_indiv_history[gen_num] = best_indiv
    best_indiv_test_perf_history[gen_num] = res

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


def _save_ppl(save_path, ppl, gen_num):
    with open(save_path / f"ppl_gen_{gen_num}.pkl", "wb") as fp:
        pickle.dump(ppl, fp)


def _save_histories(save_path, best_indiv_history,
                    best_indiv_test_perf_history):
    with open(save_path / "best_indiv_history.pkl", "wb") as fp:
        pickle.dump(best_indiv_history, fp)
    with open(save_path / "best_indiv_test_perf_history.pkl", "wb") as fp:
        pickle.dump(best_indiv_test_perf_history, fp)


def _save_main_py_script(save_path):
    main_file_path = Path(__main__.__file__)
    shutil.copy(main_file_path, save_path)


def _compress_ppl_pkl_files(save_path, num_gens):
    ppl_pkl_files = glob.glob(f"{save_path}/ppl*.pkl")
    assert len(ppl_pkl_files) == (num_gens + 1)
    # use max xz compression (9e) with as may threads as available (T0)
    os.environ["XZ_OPT"] = "-T0 -9e"
    subprocess.run(["tar", "-cJf", f"{save_path}/ppls.tar.xz"] + ppl_pkl_files,
                   check=True)


def _delete_uncompressed_ppl_pkl_files(save_path):
    ppl_pkl_files = glob.glob(f"{save_path}/ppl*.pkl")
    for file_ in ppl_pkl_files:
        os.remove(file_)


if __name__ == "__main__":
    set_start_method("spawn")
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")
