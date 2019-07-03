import math
import random

import numpy as np
import tensorflow as tf

from utils.Utilities import print_averages, get_reward_value, find_location
from utils.Visualizations import plot_episode, plot_score


class ShortestPathDeepQLearning:
    def __init__(self, maze):
        print("initializing ShortestPath - Deep Q Learning")
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

        self.num_episodes = 2000
        self.max_steps_per_episode = 100

        self.learning_rate = 0.1
        self.discount_rate = 0.95

        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.001

        # Input has five neurons, each represents single game state (0-4)
        self.input_count = self.mazeRow * self.mazeColumn
        # Output is two neurons, each represents Q-value for action (FORWARD and BACKWARD)
        self.output_count = len(self.ACTIONS)

        self.session = tf.Session()
        self.define_model()
        self.session.run(self.initializer)

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

    def take_deep_exploit_action(self, current_state):
        itr = 0
        while itr < 100:
            try:
                exploit_action = np.argmax(self.get_Q(current_state))
                rew_ = self.rewardMat[current_state, exploit_action]
                if not np.isnan(rew_):
                    return exploit_action
                itr += 1
            except ValueError as err:
                print(err)
                return np.NaN
        else:
            return np.NaN

    def take_deep_explore_action(self, current_state):
        return self.take_explore_action(current_state)

    def get_action(self, current_state):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:
            # exploit
            action = self.take_deep_exploit_action(current_state)
            if np.isnan(action):
                action = self.take_deep_explore_action(current_state)
        else:
            # explore
            action = self.take_deep_explore_action(current_state)

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
        self.train(state, action, reward, new_state)

    def train(self, old_state, action, reward, new_state):
        old_state_Q_values = self.get_Q(old_state)
        new_state_Q_values = self.get_Q(new_state)

        old_state_Q_values[action] = reward + self.discount_rate * np.amax(new_state_Q_values)

        training_input = self.to_one_hot(old_state)
        target_output = [old_state_Q_values]
        training_data = {self.model_input: training_input, self.target_output: target_output}

        self.session.run(self.optimizer, feed_dict=training_data)

    def find(self):
        self.populate_reward_matrix()

        sx, sy = find_location(self.maze, 2)
        start_state = (sx * self.mazeColumn) + sy

        rewards_all_episodes = []

        episode_steps = []
        for episode in range(self.num_episodes):
            # initialize new episode params
            if episode % 100 == 0:
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
        # plot_episode(np.matrix(self.maze), episode_steps[highest_score_index]['states'])
        plot_score(rewards_all_episodes)

    def define_model(self):
        self.model_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_count])
        fc1 = tf.layers.dense(self.model_input, 32, activation=tf.sigmoid,
                              kernel_initializer=tf.constant_initializer(np.zeros((self.input_count, 32))))

        fc2 = tf.layers.dense(fc1, 16, activation=tf.sigmoid,
                              kernel_initializer=tf.constant_initializer(np.zeros((32, 16))))

        fc3 = tf.layers.dense(fc2, 16, activation=tf.sigmoid,
                              kernel_initializer=tf.constant_initializer(np.zeros((16, self.output_count))))
        self.model_output = tf.layers.dense(fc3, self.output_count)

        self.target_output = tf.placeholder(shape=[None, self.output_count], dtype=tf.float32)

        loss = tf.losses.mean_squared_error(self.target_output, self.model_output)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        self.initializer = tf.global_variables_initializer()

    def get_Q(self, state):
        return self.session.run(self.model_output, feed_dict={self.model_input: self.to_one_hot(state)})[0]

    def to_one_hot(self, state):
        one_hot = np.zeros((1, self.input_count))
        one_hot[0, [state]] = 1
        return one_hot
