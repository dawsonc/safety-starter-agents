#!/usr/bin/env python
import gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs


def main(algo, seed):

    # Verify experiment
    algo_list = ["ppo", "ppo_lagrangian", "trpo", "trpo_lagrangian", "cpo"]
    algo = algo.lower()
    assert algo in algo_list, "Invalid algo"

    # Hyperparameters
    exp_name = algo + "_turtle2d"
    num_steps = 1e7
    steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # # Prepare Logger
    # logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # # Algo and Env
    # algo = eval("safe_rl." + algo)
    # env_name = "gym_turtle2d:turtle2d-v0"

    # algo(
    #     env_fn=lambda: gym.make(env_name),
    #     ac_kwargs=dict(
    #         hidden_sizes=(256, 256),
    #     ),
    #     epochs=epochs,
    #     steps_per_epoch=steps_per_epoch,
    #     save_freq=save_freq,
    #     target_kl=target_kl,
    #     cost_lim=cost_lim,
    #     seed=seed,
    #     logger_kwargs=logger_kwargs,
    # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.algo, args.seed)
