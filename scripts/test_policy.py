#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger

import pandas as pd


log_trace = True


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    num_collisions = 0
    num_goals_reached = 0
    total_steps_to_goal = 0

    if log_trace:
        num_episodes = 1
        trace_df = pd.DataFrame()

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if log_trace:
            log_packet = {}
            log_packet["$t$"] = 0.1 * ep_len
            log_packet["$x$"] = o[-3]
            log_packet["$y$"] = o[-2]
            trace_df = trace_df.append(log_packet, ignore_index=True)

        if info.get("in_collision", False):
            num_collisions += 1
        if info.get("goal_reached", False):
            num_goals_reached += 1
            total_steps_to_goal += ep_len

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    print("=======================")
    print("Experiment Summary")
    print("=======================")
    print(f"Number of trials in random environments: {num_episodes}")
    print(f"Max episode length (steps): {max_ep_len}")
    print(f"Number of collisions: {num_collisions}")
    print(f"Number of goals reached: {num_goals_reached}")
    if num_goals_reached > 0:
        print(f"Avg. steps to goal when reached: {total_steps_to_goal / num_goals_reached}")

    if log_trace:
        trace_df.to_csv("trace.csv", index=False)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
