from bestarm import *
import numpy as np
from random import shuffle
from blocks import successive_block_rejection
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

def test_graphs(iterations):
    """! three settings are tested: little connection ( < 0.05),
    medium connection ( < 0.2), great connection ( < 0.5) """
    graph_wins = [0, 0, 0]
    sr_wins = 0
    connection = [0.05, 0.2, 0.5]
    for k in range(iterations):
        print(k)
        mab = MABSideObservation()
        K = np.random.randint(20, 100)
        T = np.random.randint(100, 5000)
        mab.add_bernoulli_arms(np.random.rand(K))
        mab2 = MultiArmedBandit(mab)
        truth = mab.get_best_arm_index_and_expectation()[0]
        if succesive_rejection(mab2, T) == truth:
            sr_wins += 1
        for i, c in enumerate(connection):
            mab.add_graph(np.random.rand(K, K) < c)
            #print(np.sum(mab.W))
            res = successive_block_rejection(mab, T)
            if res == truth:
                graph_wins[i] += 1
    sr_wins /= float(iterations)
    for k in [0, 1, 2]:
        graph_wins[k] /= float(iterations)
    return [sr_wins] + graph_wins

if __name__ == "__main__":
    iterations = 3000
    graph_wins = test_graphs(iterations)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    wdth = .3
    ax.bar(np.arange(4)-wdth/2, graph_wins, wdth, color=(1, 128./255, 0))
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['Succ-rej', 'sparse', 'mid', 'dense'])
    ax.set_ylabel('well identified best-arm proportion')
    plt.show()
