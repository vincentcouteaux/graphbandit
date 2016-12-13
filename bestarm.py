import numpy as np

class armBernouilli:
    def __init__(self, p):
        self.p = p
    def pull(self):
        if np.random.rand() > self.p:
            return 1.
        else:
            return 0.

class armGaussian:
    def __init__(self, mu, sigma):
        self.p = mu
        self.sigma = sigma
    def pull(self):
        r = np.random.


