from gym_env import ArmyRetentionEnv
import argparse
from argparse import Namespace
import json
import logging
from train import policy_eval,policy_iteration
import numpy as np
from utils import display, get_policy

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--params_file", type=str,
                        help="JSON configuration file")
    parser.add_argument("--dataroot", type=str, help="Path to dataset.")
    parser.add_argument("--output", type=str,default=None, help="Output path")
    parser.add_argument("--simulate", action="store_true", help="Simulation")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        args.update(params)
        args = Namespace(**args)

        # for key in vars(args):
        #     logger.info(str(key) + " = " + str(vars(args)[key]))
    args.params = params  # used for saving checkpoints

    env = ArmyRetentionEnv(args)

    pol_iter_policy = policy_iteration(env,policy_eval,args.beta)
    policy = get_policy(pol_iter_policy[0],env)

    if args.output:
        display(policy,args)

    if args.simulate:
        from simulation import simulate

        simulate(env,pol_iter_policy[0])


if (__name__ == "__main__"):
    main()

