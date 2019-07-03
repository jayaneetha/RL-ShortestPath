import numpy as np

from ShortestPathDeepQlearning import ShortestPathDeepQLearning
from ShortestPathQLearning import ShortestPathQLearning

if __name__ == "__main__":
    maze = np.matrix([
        [1, 0, 1, 0],
        [2, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 4, 1],
        [0, 1, 0, 1]
    ])

    # Deep Q Learning Algorithm
    sp_dq = ShortestPathDeepQLearning(maze)
    sp_dq.find()

    # Q Table Algorithm
    # sp_q = ShortestPathQLearning(maze)
    # sp_q.find()
