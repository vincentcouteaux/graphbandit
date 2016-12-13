import numpy as np
from banditsetting import *

def general_sequential_elimination(multi_armed_bandit, budget, z, b):
    n_arms = multi_armed_bandit.get_number_of_arms()
    n_rounds  = b.size
    z=np.double(z)
    C = 1/z[-1] + np.sum(b/z)
    n = np.ceil((budget - n_arms)/(C*z))
    remaining_indices = np.arange(n_arms)
    cumulated_reward = np.zeros(n_arms)
    visits = np.zeros(n_arms)
    for r in range(n_rounds):
        if r == 0:
            t = n[0]
        else:
            t = n[r] - n[r - 1]
        t = int(t)
        for a in remaining_indices[remaining_indices != -1]:
            for k in range(t):
                cumulated_reward[a] += multi_armed_bandit.pull(a)
                visits[a] += 1
                #print('arm {0}: cumreward = {1}, visits = {2}'.format(a, cumulated_reward[a], visits[a]))
        average_reward = cumulated_reward/visits
        for k in range(int(b[r])):
            average_reward[remaining_indices == -1] = np.max(average_reward) + 1
            remaining_indices[np.argmin(average_reward)] = -1
    return np.max(remaining_indices)

def nonlinear_seq_elimination(multi_armed_bandit, budget, param):
    K = multi_armed_bandit.get_number_of_arms()
    return general_sequential_elimination(multi_armed_bandit, budget, (K - np.arange(K - 1))**param, np.ones(K-1))

def succesive_rejection(multi_armed_bandit, budget):
    return nonlinear_seq_elimination(multi_armed_bandit, budget, 1)


if __name__ == "__main__":
    mab = MultiArmedBandit()
    mab.add_bernoulli_arms([0.1, 0.005, 0.3, 0.4, 0.18, 0.7, 0.8, 0.12, 0.79, 0.23])
    print(mab.get_best_arm_index_and_expectation())
    print(mab._get_deltas())
    print(mab.getH1())
    print(mab.getH2())
    print(succesive_rejection(mab, 200))


