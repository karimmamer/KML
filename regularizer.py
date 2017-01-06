import numpy as np

class L2:

    def __init__(self, n_samples, C = 0):
        self.C = C * 1.0
        self.n_samples = n_samples * 1.0
        
    def evaluate(self, weights):
        weights[1:] = weights[1:]**2
        return self.C * weights[1:].sum()/ (2.0 * self.n_samples)
        
    def evaluate_gradient(self, weights):
        gradient = np.zeros(weights.shape)
        gradient[1:] = self.C * weights[1:] / self.n_samples
        return gradient
        
class L0:

    def __init__(self):
        pass
        
    def evaluate(self, weights):
        return 0
        
    def evaluate_gradient(self, weights):
        return np.zeros(weights.shape)
