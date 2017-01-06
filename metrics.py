import numpy as np

class MSE:

    def __init__(self, X, y, h):
        self.X = X
        self.y = y
        self.h = h
        
    def evaluate(self, weights):
        error = (self.h(weights, self.X) - self.y)**2
        return error.mean() * 0.5
        
    def evaluate_gradient(self, weights):
        gradient_error = np.dot(self.X.T, self.h(weights, self.X) - self.y) / self.X.shape[0]
        return gradient_error
        
class LogLoss:

    def __init__(self, X, y, h):
        self.X = X
        self.y = y
        self.h = h
        
    def evaluate(self, weights):
        error = -1.0 * self.y * np.log(self.h(weights, self.X)) - (1.0 - self.y) * np.log(1.0 - self.h(weights, self.X))
        return error.sum() / self.X.shape[0]
        
    def evaluate_gradient(self, weights):
        gradient_error = np.dot(self.X.T, self.h(weights, self.X) - self.y) / self.X.shape[0]
        return gradient_error
