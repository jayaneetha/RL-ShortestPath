import random

import numpy as np
import matplotlib.pyplot as plt

from utils.Utilities import find_location, get_reward_value, take_exploit_action, take_explore_action, take_action, \
    print_averages
from utils.Visualizations import plot_episode, plot_score

goal_mark = 4
start_mark = 2
rewards = np.array([-1, -2, -3, -1])

maze = np.matrix([
    [1, 0, 1, 0],
    [2, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 4, 1],
    [0, 1, 0, 1]
])
# maze = np.matrix([
#     [1 ,0, 0, 1, 1],
#     [0, 2, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 0],
#     [1, 0, 1, 1, 0],
#     [1, 1, 1, 1, 4]
# ])

# maze = np.matrix([
#     [1 ,0, 0, 1, 1, 0],
#     [0, 2, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 1, 1, 0, 0],
#     [1, 0, 0, 1, 0, 0],
#     [1, 1, 1, 1, 1, 1],
#     [1, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 4]
# ])
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


def populate_reward_matrix():
    print("Population Reward Matrix")
    for i in range(0, mazeRow):
        for j in range(0, mazeColumn):
            stateIterator_ = (i * mazeColumn) + j  # Defining Current State
            if i > 0:
                # top row
                r = get_reward_value(maze[i, j], maze[i - 1, j])
                if maze[i - 1, j] == goal_mark:
                    r = rewForGoal
                rewardMat[stateIterator_, actUp] = r

            if i < mazeRow - 1:
                # bottom row
                r = get_reward_value(maze[i, j], maze[i + 1, j])
                if maze[i + 1, j] == goal_mark:
                    r = rewForGoal
                rewardMat[stateIterator_, actDown] = r

            if j > 0:
                # left row
                r = get_reward_value(maze[i, j], maze[i, j - 1])
                if maze[i, j - 1] == goal_mark:
                    r = rewForGoal
                rewardMat[stateIterator_, actLeft] = r

            if j < mazeColumn - 1:
                # right row
                r = get_reward_value(maze[i, j], maze[i, j + 1])
                if maze[i, j + 1] == goal_mark:
                    r = rewForGoal
                rewardMat[stateIterator_, actRight] = r


if __name__ == "__main__":

    populate_reward_matrix()

    sx, sy = find_location(maze, 2)
    start_state = (sx * mazeColumn) + sy

    gx, gy = find_location(maze, 4)
    goal_state = (gx * mazeColumn) + gy

    rewards_all_episodes = []
    # Q-learning algorithm
    episode_steps = []
    for episode in range(num_episodes):
        # initialize new episode params
        if episode % 1000 == 0:
            print("Training Episode {}".format(episode))

        state = start_state
        done = False
        rewards_current_episode = 0

        step_actions = []
        step_states = [state]
        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                # exploit
                action = take_exploit_action(rewardMat, Q, state)
                if np.isnan(action):
                    action = take_explore_action(rewardMat, state)
            else:
                # explore
                action = take_explore_action(rewardMat, state)

            # Take new action
            new_state, reward, done = take_action(rewardMat, maze, state, action, goal_mark)

            # Update Q-table
            q_value = Q[state, action]
            q_ = q_value * (1 - learning_rate) + (learning_rate * (reward + discount_rate * np.max(Q[new_state, :])))
            Q[state, action] = q_

            # Set new state
            state = new_state
            step_actions.append(action)
            step_states.append(state)

            # Add new reward
            rewards_current_episode += reward

            if done:
                break

        episode_steps.append({"actions": step_actions, "states": step_states})
        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * episode)
        # Add current episode reward to total rewards list
        rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    print_averages(rewards_all_episodes, num_episodes)

    # Visualizations
    highest_score_index = np.argmax(rewards_all_episodes)
    print("Best Episode: {}".format(highest_score_index))
    plot_episode(np.matrix(maze), episode_steps[highest_score_index]['states'])
    plot_score(rewards_all_episodes)
