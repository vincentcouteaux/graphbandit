import numpy as np

class armBernoulli:
    def __init__(self, p):
        self.p = p
    def pull(self):
        if np.random.rand() > self.p:
            return 0.
        else:
            return 1.

class armGaussian:
    def __init__(self, mu, sigma):
        self.p = mu
        self.sigma = sigma
    def pull(self):
        r = np.random.randn()*self.sigma + self.mu
        if r > 1.:
            return 1.
        elif r < 0.:
            return 0.
        else:
            return r

def generate_bernoulli_mab(averages):
    """ generate a multi-armed bandit, with bernoulli arms
    @param averages list of averages of the arms
    @return a list of armBernoulli, with given expectations
    """
    mab = []
    for a in averages:
        mab.append(armBernoulli(a))
    return mab

def general_sequential_elimination(multi_armed_bandit, budget, z, b):
    n_arms = len(multi_armed_bandit)
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
                cumulated_reward[a] += multi_armed_bandit[a].pull()
                visits[a] += 1
                #print('arm {0}: cumreward = {1}, visits = {2}'.format(a, cumulated_reward[a], visits[a]))
        average_reward = cumulated_reward/visits
        for k in range(int(b[r])):
            average_reward[remaining_indices == -1] = np.max(average_reward) + 1
            remaining_indices[np.argmin(average_reward)] = -1
    return np.max(remaining_indices)

def nonlinear_seq_elimination(multi_armed_bandit, budget, param):
    K = len(multi_armed_bandit)
    return general_sequential_elimination(multi_armed_bandit, budget, (K - np.arange(K - 1))**param, np.ones(K-1))

def succesive_rejection(multi_armed_bandit, budget):
    return nonlinear_seq_elimination(multiArmedBandit, budget, 1)


if __name__ == "__main__":
    multiArmedBandit = (armBernoulli(0.3), armBernoulli(0.15), armBernoulli(0.14), armBernoulli(0.1))
    print(succesive_rejection(multiArmedBandit, 200))
    multiArmedBandit = generate_bernoulli_mab([0.1, 0.005, 0.3, 0.4, 0.18, 0.7, 0.8, 0.12, 0.79, 0.23])
    print(multiArmedBandit[succesive_rejection(multiArmedBandit, 200)].p)


