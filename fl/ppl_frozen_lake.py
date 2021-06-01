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
from ppl.inference import DecisionListInference, SpecificityInference
from ppl.ppl import PPL
from rlenvs.environment import assess_perf
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_FL_SEED = 0
_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--fl-grid-size", type=int, required=True)
    parser.add_argument("--fl-slip-prob", type=float, required=True)
    parser.add_argument("--ppl-num-gens", type=int, required=True)
    parser.add_argument("--ppl-seed", type=int, required=True)
    parser.add_argument("--ppl-pop-size", type=int, required=True)
    parser.add_argument("--ppl-indiv-size-min", type=int, required=True)
    parser.add_argument("--ppl-indiv-size-max", type=int, required=True)
    parser.add_argument("--ppl-inference-strat",
                        choices=["sp", "dl"],
                        required=True)
    parser.add_argument("--ppl-default-action", type=int, required=True)
    parser.add_argument("--ppl-num-elites", type=int, required=True)
    parser.add_argument("--ppl-tourn-size", type=int, required=True)
    parser.add_argument("--ppl-p-cross", type=float, required=True)
    parser.add_argument("--ppl-p-mut", type=float, required=True)
    parser.add_argument("--ppl-m-nought", type=int, required=True)
    parser.add_argument("--num-train-rollouts", type=int, required=True)
    parser.add_argument("--num-test-rollouts", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    train_env = _make_train_env(args)
    test_env = _make_test_env(args)
    ppl_hyperparams = {
        "seed": args.ppl_seed,
        "pop_size": args.ppl_pop_size,
        "indiv_size_min": args.ppl_indiv_size_min,
        "indiv_size_max": args.ppl_indiv_size_max,
        "num_elites": args.ppl_num_elites,
        "tourn_size": args.ppl_tourn_size,
        "p_cross": args.ppl_p_cross,
        "p_mut": args.ppl_p_mut,
        "m_nought": args.ppl_m_nought,
        "num_rollouts": args.num_train_rollouts,
        "gamma": args.gamma
    }
    encoding = IntegerUnorderedBoundEncoding(train_env.obs_space)
    inference_strat = _make_inference_strat(args, train_env)
    ppl = PPL(train_env,
              encoding,
              inference_strat,
              hyperparams_dict=ppl_hyperparams)

    pop_history = {}
    init_pop = ppl.init()
    pop_history[0] = init_pop
    _log_pop_stats(0, init_pop, test_env, args)
    for gen_num in range(1, args.ppl_num_gens + 1):
        pop = ppl.run_gen()
        _log_pop_stats(gen_num, pop, test_env, args)
        pop_history[gen_num] = pop

    _save_data(save_path, pop_history)
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
        iod_strat = "frozen_no_repeat"
    elif slip_prob > 0:
        iod_strat = "goal_dist"
    else:
        assert False
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=slip_prob,
                   iod_strat=iod_strat,
                   seed=_FL_SEED)


def _make_test_env(args):
    slip_prob = args.fl_slip_prob
    if slip_prob == 0:
        iod_strat = "frozen_no_repeat"
    elif slip_prob > 0:
        iod_strat = "frozen_repeat"
    else:
        assert False
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=slip_prob,
                   iod_strat=iod_strat,
                   seed=_FL_SEED)


def _make_inference_strat(args, env):
    if args.ppl_inference_strat == "sp":
        cls = SpecificityInference
    elif args.ppl_inference_strat == "dl":
        cls = DecisionListInference
    else:
        assert False
    return cls(action_space=env.action_space,
               default_action=args.ppl_default_action)


def _log_pop_stats(gen_num, pop, test_env, args):
    logging.info(f"gen num {gen_num}")
    fitnesses = [indiv.fitness for indiv in pop]
    min_ = np.min(fitnesses)
    mean = np.mean(fitnesses)
    max_ = np.max(fitnesses)
    logging.info(f"min, mean, max fitness in pop: {min_}, {mean}, {max_}")

    non_elites = [indiv for indiv in pop if not indiv.is_elite]
    total_time_steps_used = sum(
        [indiv.time_steps_used for indiv in non_elites])
    logging.info(f"total time steps used: {total_time_steps_used}")

    # find out actual perf of max fitness indiv
    best_indiv = sorted(pop, key=lambda indiv: indiv.fitness, reverse=True)[0]
    res = assess_perf(test_env, best_indiv, args.num_test_rollouts, args.gamma)
    if res.failed:
        logging.info("best failed actual perf assessment")
    logging.info(f"actual best perf: {res.perf}")


def _save_data(save_path, pop_history):
    with open(save_path / "pop_history.pkl", "wb") as fp:
        pickle.dump(pop_history, fp)


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
