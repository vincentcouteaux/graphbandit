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

if __name__ == "__main__":
    mab = MABSideObservation()
    mab.add_bernoulli_arms(np.random.rand(50))
    mab.add_graph(np.random.randn(50, 50))
    #print(mab.pull(3))
    #print(mab.pull(10))
    #print(mab.pull(3))
    print(get_blocks(mab.W))
