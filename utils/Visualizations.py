import matplotlib.pyplot as plt
import math

import numpy as np
from matplotlib import animation


def plot_score(rewards_all_episodes):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(rewards_all_episodes, 'r')
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


def get_frame(maze, state):
    mazeColumn = maze.shape[1]
    mx = math.floor(state / mazeColumn)
    my = state % mazeColumn
    maze[mx, my] = 10
    return maze


def plot_episode(maze, states):
    frames = []
    for s in states:
        m = get_frame(np.matrix(maze), s)
        frames.append(m)

    def update(i):
        matrice.set_array(frames[i])

    fig, ax = plt.subplots()
    matrice = ax.matshow(frames[0])

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=500, repeat_delay=50)
    plt.show()
