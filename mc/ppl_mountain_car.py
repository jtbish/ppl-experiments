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

import numpy as np
from ppl.encoding import RealUnorderedBoundEncoding
from ppl.inference import DecisionListInference, SpecificityInference
from ppl.ppl import PPL
from rlenvs.mountain_car import make_mountain_car_env as make_mc
import __main__

_MC_IOD_STRAT = "bottom_zero_vel"
_MC_SEED = 0
_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--ppl-num-gens", type=int, required=True)
    parser.add_argument("--ppl-seed", type=int, required=True)
    parser.add_argument("--ppl-pop-size", type=int, required=True)
    parser.add_argument("--ppl-indiv-size-min", type=int, required=True)
    parser.add_argument("--ppl-indiv-size-max", type=int, required=True)
    parser.add_argument("--ppl-inference-strat",
                        choices=["sp", "dl"],
                        required=True)
    parser.add_argument("--ppl-num-elites", type=int, required=True)
    parser.add_argument("--ppl-tourn-size", type=int, required=True)
    parser.add_argument("--ppl-p-mut", type=float, required=True)
    parser.add_argument("--ppl-m-nought", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--num-rollouts", type=int, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    env = _make_env()
    ppl_hyperparams = {
        "seed": args.ppl_seed,
        "pop_size": args.ppl_pop_size,
        "indiv_size_min": args.ppl_indiv_size_min,
        "indiv_size_max": args.ppl_indiv_size_max,
        "num_elites": args.ppl_num_elites,
        "tourn_size": args.ppl_tourn_size,
        "p_mut": args.ppl_p_mut,
        "m_nought": args.ppl_m_nought,
        "gamma": args.gamma,
        "num_rollouts": args.num_rollouts
    }
    encoding = RealUnorderedBoundEncoding(env.obs_space)
    inference_strat = _make_inference_strat(args, env)
    ppl = PPL(env, encoding, inference_strat, hyperparams_dict=ppl_hyperparams)

    gen_history = {}
    init_pop = ppl.init()
    gen_history[0] = init_pop
    _log_pop_stats(0, init_pop)
    for gen_num in range(1, args.ppl_num_gens + 1):
        next_pop = ppl.run_gen()
        _log_pop_stats(gen_num, next_pop)
        gen_history[gen_num] = next_pop

    _save_data(save_path, gen_history)
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


def _make_env():
    return make_mc(_MC_IOD_STRAT, _MC_SEED)


def _make_inference_strat(args, env):
    if args.ppl_inference_strat == "sp":
        return SpecificityInference(env.action_space)
    elif args.ppl_inference_strat == "dl":
        return DecisionListInference(env.action_space)
    else:
        assert False


def _log_pop_stats(gen_num, pop):
    logging.info(f"gen num {gen_num}")
    fitnesses = [indiv.fitness for indiv in pop]
    min_ = np.min(fitnesses)
    mean = np.mean(fitnesses)
    max_ = np.max(fitnesses)
    logging.info(f"min, mean, max fitness in pop: {min_}, {mean}, {max_}")


def _save_data(save_path, gen_history):
    with open(save_path / "gen_history.pkl", "wb") as fp:
        pickle.dump(gen_history, fp)


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
