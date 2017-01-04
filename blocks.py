import numpy as np
from banditsetting import *

def get_blocks(adjacency):
    """! @param adjacency the adjacency matrix"""
    W = np.array(adjacency)
    blocks={}
    visited = np.zeros(W.shape[0], dtype=bool)
    while W.any():
        most_connected = np.argmax(np.sum(W, 0))
        blocks[most_connected] = W[most_connected, :].nonzero()[0]
        visited[most_connected] = True
        for n in blocks[most_connected]:
            visited[n] = True
            for i in range(W.shape[0]):
                W[i, n] = False
                W[n, i] = False
    for i in np.logical_not(visited).nonzero()[0]:
        blocks[i] = []
    return blocks

def sequential_block_elimination(mab_so, T, z, blocks):
    G = np.array([r for r in blocks])
    M = len(blocks)
    C = np.sum(1./z)
    n = np.ceil(float(T-M)/(C*z))
    cumulated_reward = np.zeros(mab_so.get_number_of_arms())
    visits = np.zeros(mab_so.get_number_of_arms())
    pulls = 0
    for r in range(M-1):
        if r == 0:
            t = n[0]
        else:
            t = n[r] - n[r - 1]
        t = int(t)
        for v in G[G != -1]:
            for k in range(t):
                r, observs = mab_so.pull(v)
                pulls += 1
                cumulated_reward[v] += r
                visits[v] += 1
                for a in blocks[v]:
                    cumulated_reward[a] += observs[a]
                    visits[a] += 1
        y = np.zeros(M) + np.max(visits)
        for i, v in enumerate(G):
            if v != -1:
                current_max = 0.
                for a in blocks[v]:
                    if cumulated_reward[a]/visits[a] >= current_max:
                        current_max = cumulated_reward[a]/visits[a]
                y[i] = current_max
        G[np.argmin(y)] = -1
    #The best block is identified
    best_block = np.max(G)
    if len(n) > 1:
        t = int(n[-1] - n[-2])
    else:
        t = int(n[0])
    for k in range(t):
        r, observs = mab_so.pull(best_block)
        pulls += 1
        cumulated_reward[best_block] += r
        visits[best_block] += 1
        for a in blocks[best_block]:
            cumulated_reward[a] += observs[a]
            visits[a] += 1
    maxi, argmax = cumulated_reward[best_block]/visits[best_block], best_block
    for a in blocks[best_block]:
        if cumulated_reward[a]/visits[a] > maxi:
            maxi = cumulated_reward[a]/visits[a]
            argmax = a
    if pulls > T:
        print('n= {}, budget = {}, pulls = {}'.format(n, T, pulls))
    return argmax


def successive_block_rejection(mab_so, T):
    blocks = get_blocks(mab_so.W)
    M = len(blocks)
    return sequential_block_elimination(mab_so, T, M+1 - np.arange(M), blocks)


if __name__ == "__main__":
    mab = MABSideObservation()
    K = 50
    mab.add_bernoulli_arms(np.random.rand(K))
    mab.add_graph(np.random.randn(K, K))
    #print(mab.pull(3))
    #print(mab.pull(10))
    #print(mab.pull(3))
    print(get_blocks(mab.W))
    best_guess = successive_block_rejection(mab, 2000)
    print('guess: arm {} with actual mean {}'.format(best_guess, mab._arms[best_guess].p))
    real_best, avg = mab.get_best_arm_index_and_expectation()
    print('truth: arm {} with actual mean {}'.format(real_best, avg))
