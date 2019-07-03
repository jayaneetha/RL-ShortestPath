import numpy as np


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


def find_location(maze, number):
    _r = np.where(maze == number)[0][0]
    _c = np.where(maze == number)[1][0]
    return _r, _c


def print_averages(rewards_all_episodes, num_episodes):
    # Calculate and print the average reward per thousand episodes
    rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
    count = 1000

    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thosand_episodes:
        print(count, ": ", str(sum(r / 1000)))
        count += 1000
