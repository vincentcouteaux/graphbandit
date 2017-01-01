import numpy as np

class ArmBernoulli:
    def __init__(self, p):
        self.p = p
    def pull(self):
        if np.random.rand() > self.p:
            return 0.
        else:
            return 1.

class ArmGaussian:
    def __init__(self, mu, sigma):
        self.p = mu
        self.sigma = sigma
    def pull(self):
        r = np.random.randn()*self.sigma + self.p
        return r

class ArmBeta:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.p = float(a)/(a+b)
    def pull(self):
        return np.random.beta(self.a, self.b)

class MultiArmedBandit:
    def __init__(self):
        self._arms = []

    def add_bernoulli_arms(self, averages):
        for a in averages:
            self._arms.append(ArmBernoulli(a))

    def add_gaussian_arms(self, averages, stds):
        for i, a in enumerate(averages):
            self._arms.append(ArmGaussian(a, stds[i]))

    def add_beta_arms(self, param_a, param_b):
        for i, a in enumerate(param_a):
            self._arms.append(ArmBeta(a, param_b[i]))

    def get_number_of_arms(self):
        return len(self._arms)

    def pull(self, index):
        return self._arms[index].pull()

    def get_best_arm_index_and_expectation(self):
        index = 0
        expectation = self._arms[0]
        for i, arm in enumerate(self._arms):
            if arm.p > expectation:
                index = i
                expectation = arm.p
        return (index, expectation)

    def _get_deltas(self):
        istar, pstar = self.get_best_arm_index_and_expectation()
        deltas = np.zeros(len(self._arms))
        for i, arm in enumerate(self._arms):
            deltas[i] = pstar - arm.p
        return deltas

    def getH1(self):
        deltas = self._get_deltas()
        return np.sum(1./(deltas[deltas != 0]**2))

    def getH(self, p):
        deltas = self._get_deltas()
        order = np.argsort(np.argsort(deltas)) + 1 # rank of each arm (1 is optimal, 2 is second best...) Je suis assez fier de la combine argsort(argsort())...
        return np.max((order[order != 1]**p)/(deltas[order != 1]**2))

    def getH2(self):
        return self.getH(1)

class MABSideObservation(MultiArmedBandit):
    def add_graph(self, W):
        """! @param W the adjacency matrix """
        self.W = W
        self.W = self.W > 0.5
        self.W = np.logical_or(self.W, self.W.T)
        for i in range(W.shape[0]):
            self.W[i, i] = False

    def pull(self, index):
        neighbors = self.W[index, :].nonzero()[0]
        side_observations = {}
        for n in neighbors:
            side_observations[n] = self._arms[n].pull()
        return self._arms[index].pull(), side_observations

