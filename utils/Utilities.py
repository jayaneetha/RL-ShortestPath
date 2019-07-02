import numpy as np
import random
import math


def find_location(maze, number):
    _r = np.where(maze == number)[0][0]
    _c = np.where(maze == number)[1][0]
    return _r, _c


def get_reward_value(n1, n2):
    if n1 == 4 or n1 == 2:
        n1 = 0
    if n2 == 4 or n2 == 2:
        n2 = 0
    if n1 == 0 and n2 == 0:  # p-p
        return -1
    if n1 == 0 and n2 == 1:  # p-h
        return -3
    if n1 == 1 and n2 == 0:  # h-p
        return -1
    if n1 == 1 and n2 == 1:  # h-h
        return -2
    return 0


def take_explore_action(reward_mat, current_state):
    while True:
        random_action = random.randint(0, 3)
        rew_ = reward_mat[current_state, random_action]
        if not np.isnan(rew_):
            return random_action


def do_take_exploit_action(reward_mat, current_state, Q_):
    while True:
        try:
            exploit_action = np.nanargmax(Q_)
            rew_ = reward_mat[current_state, exploit_action]
            if np.isnan(rew_):
                Q_[exploit_action] = np.NAN
                return do_take_exploit_action(reward_mat, current_state, Q_)
            else:
                return exploit_action
        except ValueError as err:
            print(err)
            return np.NaN


def take_exploit_action(reward_mat, Q, current_state):
    return do_take_exploit_action(reward_mat, current_state, np.array(Q[current_state, :]))


def take_action(reward_mat, maze, current_state, action_, goal_mark):
    mazeColumn = maze.shape[1]
    new_state_ = 0
    if action_ == 0:
        new_state_ = current_state - 1
    if action_ == 1:
        new_state_ = current_state + 1
    if action_ == 2:
        new_state_ = current_state - mazeColumn
    if action_ == 3:
        new_state_ = current_state + mazeColumn

    rew_ = reward_mat[current_state, action_]

    done_ = False
    mx = math.floor(new_state_ / mazeColumn)
    my = new_state_ % mazeColumn

    if maze[mx, my] == goal_mark:
        done_ = True

    return new_state_, rew_, done_


def print_averages(rewards_all_episodes, num_episodes):
    # Calculate and print the average reward per thousand episodes
    rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
    count = 1000

    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thosand_episodes:
        print(count, ": ", str(sum(r / 1000)))
        count += 1000
