from bestarm import *
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

def alpha_bandit(alpha, K, mu1):
    """ Generate a multi-armed bandit with K arms where
    \Delta_i = 1/4 * (i/K)^\alpha for all i in {2, ..., K} """
    mab = MultiArmedBandit()
    means = mu1 - ((0.25*np.arange(2, K+1))/K)**alpha
    means = np.append(means, mu1)
    shuffle(means)
    mab.add_bernoulli_arms(means)
    return mab

def best_p(iterations):
    ps = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 7, 10]
    wins = np.zeros(len(ps))
    for k in range(iterations):
        alpha = 2*np.random.rand() + 0.25
        K = np.random.randint(3, 100)
        mu1 = 0.7*np.random.rand() + 0.3
        T = np.random.randint(2000, 100000)
        mab = alpha_bandit(alpha, K, mu1)
        print(k)
        for i, p in enumerate(ps):
            res = nonlinear_seq_elimination(mab, T, p*alpha)
            if res == mab.get_best_arm_index_and_expectation()[0]:
                wins[i] += 1
    return wins, ps

if __name__ == "__main__":
    iterations = 10000
    wins, ps = best_p(iterations)
    wins = wins/float(iterations)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(len(wins)), wins, .3, color=(10./255, 170./255, 100./255))
    ax.set_xticks(np.arange(len(wins) + .15))
    ax.set_xticklabels(ps)
    ax.set_xlabel('p / alpha')
    ax.set_ylabel('well identified best-arm proportion')
    plt.show()
