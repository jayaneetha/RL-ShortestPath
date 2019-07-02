import math
import random

import numpy as np

goal_mark = 4
start_mark = 2
rewards = np.array([-1, -2, -3, -1])

# maze = np.matrix([
#     [1 ,0, 0, 1, 1],
#     [0, 2, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 0],
#     [1, 0, 1, 1, 0],
#     [1, 1, 1, 1, 4]
# ])
maze = np.matrix([
    [1, 0, 1, 0],
    [2, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 4, 1],
    [0, 1, 0, 1]
])

actUp = 2
actDown = 3
actLeft = 0
actRight = 1
ACTIONS = [actLeft, actRight, actUp, actDown]

mazeRow = maze.shape[0]
mazeColumn = maze.shape[1]

rewardMat = np.empty((mazeRow * mazeColumn, len(ACTIONS)))
rewardMat[:] = np.NaN

rew = 0
rewForGoal = 10

Q = np.zeros((mazeRow * mazeColumn, len(ACTIONS)))

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


def get_rew_value(n1, n2):
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


def find_location(number):
    _r = np.where(maze == number)[0][0]
    _c = np.where(maze == number)[1][0]
    return _r, _c


def take_explore_action(current_state):
    while True:
        random_action = random.randint(0, 3)
        rew_ = rewardMat[current_state, random_action]
        if not np.isnan(rew_):
            return random_action


def do_take_exploit_action(current_state, Q_):
    while True:
        try:
            exploit_action = np.nanargmax(Q_)
            rew_ = rewardMat[current_state, exploit_action]
            if np.isnan(rew_):
                Q_[exploit_action] = np.NAN
                return do_take_exploit_action(current_state, Q_)
            else:
                return exploit_action
        except ValueError as err:
            print(err)
            return np.NaN


def take_exploit_action(current_state):
    return do_take_exploit_action(current_state, np.array(Q[current_state, :]))


def take_action(current_state, a):
    new_state_ = 0
    if a == 0:
        new_state_ = current_state - 1
    if a == 1:
        new_state_ = current_state + 1
    if a == 2:
        new_state_ = current_state - mazeColumn
    if a == 3:
        new_state_ = current_state + mazeColumn

    rew_ = rewardMat[current_state, a]

    done_ = False
    mx = math.floor(new_state_ / mazeColumn)
    my = new_state_ % mazeColumn

    if maze[mx, my] == goal_mark:
        done_ = True

    return new_state_, rew_, done_


def populate_reward_matrix():
    for i in range(0, mazeRow):
        for j in range(0, mazeColumn):
            stateIterator_ = (i * mazeColumn) + j  # Defining Current State
            if i > 0:
                # top row
                r = get_rew_value(maze[i, j], maze[i - 1, j])
                if maze[i - 1, j] == goal_mark:
                    r = rewForGoal
                rewardMat[stateIterator_, actUp] = r

            if i < mazeRow - 1:
                # bottom row
                r = get_rew_value(maze[i, j], maze[i + 1, j])
                if maze[i + 1, j] == goal_mark:
                    r = rewForGoal
                rewardMat[stateIterator_, actDown] = r

            if j > 0:
                # left row
                r = get_rew_value(maze[i, j], maze[i, j - 1])
                if maze[i, j - 1] == goal_mark:
                    r = rewForGoal
                rewardMat[stateIterator_, actLeft] = r

            if j < mazeColumn - 1:
                # right row
                r = get_rew_value(maze[i, j], maze[i, j + 1])
                if maze[i, j + 1] == goal_mark:
                    r = rewForGoal
                rewardMat[stateIterator_, actRight] = r


if __name__ == "__main__":

    populate_reward_matrix()

    sx, sy = find_location(2)
    start_state = (sx * mazeColumn) + sy

    gx, gy = find_location(4)
    goal_state = (gx * mazeColumn) + gy

    rewards_all_episodes = []
    # Q-learning algorithm
    episode_steps = []
    for episode in range(num_episodes):
        if episode > 9998:
            print(episode)
        # initialize new episode params
        state = start_state
        done = False
        rewards_current_episode = 0

        step_actions = []
        step_states = []
        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                # exploit
                action = take_exploit_action(state)
                if np.isnan(action):
                    action = take_explore_action(state)
            else:
                # explore
                action = take_explore_action(state)

            # print(state)

            # Take new action
            new_state, reward, done = take_action(state, action)

            # Update Q-table
            q_value = Q[state, action]
            q_ = q_value * (1 - learning_rate) + (learning_rate * (reward + discount_rate * np.max(Q[new_state, :])))
            Q[state, action] = q_

            # Set new state
            state = new_state

            # Add new reward
            rewards_current_episode += reward

            if done:
                break
        episode_steps.append(step_actions)
        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * episode)
        # Add current episode reward to total rewards list
        rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
    count = 1000

    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thosand_episodes:
        print(count, ": ", str(sum(r / 1000)))
        count += 1000
