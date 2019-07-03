import math
import random

import numpy as np

from utils.Utilities import print_averages, get_reward_value, find_location
from utils.Visualizations import plot_episode, plot_score


class ShortestPathQLearning:
    def __init__(self, maze):
        print("initializing ShortestPath - Q Learning")
        self.goal_mark = 4
        self.start_mark = 2
        self.rewards = np.array([-1, -2, -3, -1])
        self.maze = maze
        self.actUp = 2
        self.actDown = 3
        self.actLeft = 0
        self.actRight = 1
        self.ACTIONS = [self.actLeft, self.actRight, self.actUp, self.actDown]

        self.mazeRow = self.maze.shape[0]
        self.mazeColumn = self.maze.shape[1]
        self.rewardMat = np.empty((self.mazeRow * self.mazeColumn, len(self.ACTIONS)))
        self.rewardMat[:] = np.NaN

        self.rew = 0
        self.rewForGoal = 10

        self.Q = np.zeros((self.mazeRow * self.mazeColumn, len(self.ACTIONS)))

        self.num_episodes = 10000
        self.max_steps_per_episode = 100

        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.001

    def populate_reward_matrix(self):
        print("Populating Reward Matrix")
        for i in range(0, self.mazeRow):
            for j in range(0, self.mazeColumn):
                stateIterator_ = (i * self.mazeColumn) + j  # Defining Current State
                if i > 0:
                    # top row
                    r = get_reward_value(self.maze[i, j], self.maze[i - 1, j])
                    if self.maze[i - 1, j] == self.goal_mark:
                        r = self.rewForGoal
                    self.rewardMat[stateIterator_, self.actUp] = r

                if i < self.mazeRow - 1:
                    # bottom row
                    r = get_reward_value(self.maze[i, j], self.maze[i + 1, j])
                    if self.maze[i + 1, j] == self.goal_mark:
                        r = self.rewForGoal
                    self.rewardMat[stateIterator_, self.actDown] = r

                if j > 0:
                    # left row
                    r = get_reward_value(self.maze[i, j], self.maze[i, j - 1])
                    if self.maze[i, j - 1] == self.goal_mark:
                        r = self.rewForGoal
                    self.rewardMat[stateIterator_, self.actLeft] = r

                if j < self.mazeColumn - 1:
                    # right row
                    r = get_reward_value(self.maze[i, j], self.maze[i, j + 1])
                    if self.maze[i, j + 1] == self.goal_mark:
                        r = self.rewForGoal
                    self.rewardMat[stateIterator_, self.actRight] = r

    def take_explore_action(self, current_state):
        while True:
            random_action = random.randint(0, 3)
            rew_ = self.rewardMat[current_state, random_action]
            if not np.isnan(rew_):
                return random_action

    def do_take_exploit_action(self, current_state, Q_):
        while True:
            try:
                exploit_action = np.nanargmax(Q_)
                rew_ = self.rewardMat[current_state, exploit_action]
                if np.isnan(rew_):
                    Q_[exploit_action] = np.NAN
                    return self.do_take_exploit_action(current_state, Q_)
                else:
                    return exploit_action
            except ValueError as err:
                print(err)
                return np.NaN

    def take_exploit_action(self, current_state):
        return self.do_take_exploit_action(current_state, np.array(self.Q[current_state, :]))

    def get_action(self, current_state):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:
            # exploit
            action = self.take_exploit_action(current_state)
            if np.isnan(action):
                action = self.take_explore_action(current_state)
        else:
            # explore
            action = self.take_explore_action(current_state)

        return action

    def take_action(self, current_state, action_):
        mazeColumn = self.maze.shape[1]
        new_state_ = 0
        if action_ == 0:
            new_state_ = current_state - 1
        if action_ == 1:
            new_state_ = current_state + 1
        if action_ == 2:
            new_state_ = current_state - mazeColumn
        if action_ == 3:
            new_state_ = current_state + mazeColumn

        rew_ = self.rewardMat[current_state, action_]

        done_ = False
        mx = math.floor(new_state_ / mazeColumn)
        my = new_state_ % mazeColumn

        if self.maze[mx, my] == self.goal_mark:
            done_ = True

        return new_state_, rew_, done_

    def update(self, state, action, reward, new_state):
        q_value = self.Q[state, action]
        q_ = q_value * (1 - self.learning_rate) + (
                self.learning_rate * (reward + self.discount_rate * np.max(self.Q[new_state, :])))
        self.Q[state, action] = q_

    def find(self):
        self.populate_reward_matrix()

        sx, sy = find_location(self.maze,2)
        start_state = (sx * self.mazeColumn) + sy

        # gx, gy = find_location(4)
        # goal_state = (gx * self.mazeColumn) + gy

        rewards_all_episodes = []
        # Q-learning algorithm
        episode_steps = []
        for episode in range(self.num_episodes):
            # initialize new episode params
            if episode % 1000 == 0:
                print("Training Episode {}".format(episode))

            state = start_state
            done = False
            rewards_current_episode = 0

            step_actions = []
            step_states = [state]
            for step in range(self.max_steps_per_episode):

                # Exploration-exploitation trade-off
                action = self.get_action(state)

                # Take new action
                new_state, reward, done = self.take_action(state, action)

                # Update Q-table
                self.update(state, action, reward, new_state)

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
            self.exploration_rate = self.min_exploration_rate + (
                    self.max_exploration_rate - self.min_exploration_rate) * np.exp(
                -self.exploration_decay_rate * episode)
            # Add current episode reward to total rewards list
            rewards_all_episodes.append(rewards_current_episode)

        # Calculate and print the average reward per thousand episodes
        print_averages(rewards_all_episodes, self.num_episodes)

        # Visualizations
        highest_score_index = np.argmax(rewards_all_episodes)
        print("Best Episode: {}".format(highest_score_index))
        plot_episode(np.matrix(self.maze), episode_steps[highest_score_index]['states'])
        plot_score(rewards_all_episodes)
