import numpy as np
from scipy.optimize import fmin_bfgs

class BatchGradientDescent:
    
    def __init__(self, dim, learning_rate, error_function, regularizer):
        self.dim = dim
        self.learning_rate = learning_rate
        self.error_function = error_function 
        self.regularizer = regularizer
        self.weights = np.zeros(dim)
        self.first_time = True
        
    def minimize(self, max_iter = 1500, warm_start = False, verbose = False):
        if self.first_time or not warm_start:
            self.weights = np.zeros(self.dim)
            self.first_time = False
        
        for i in range(max_iter):
            if verbose:
                print self.error_function.evaluate(self.weights) + self.regularizer.evaluate(self.weights)
            self.weights -= self.learning_rate * (self.error_function.evaluate_gradient(self.weights) + self.regularizer.evaluate_gradient(self.weights))
            
        if verbose:
            print self.error_function.evaluate(self.weights) + self.regularizer.evaluate(self.weights)
            
        return self.weights


        
class LinearSolver:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def solve(self):
        return np.dot(np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T), self.y)



class BFGS:

    def __init__(self, dim, error_function, regularizer):
        self.dim = dim
        self.error_function = error_function 
        self.regularizer = regularizer
        self.weights = np.zeros(dim)
        
    def minimize(self, max_iter = 400, verbose = False):
        self.verbose = verbose
        return fmin_bfgs(self._evaluate_error, self.weights, fprime = self._evaluate_gradient, maxiter=max_iter)
        
    def _evaluate_error(self, weights):
        self.weights = weights
        error = self.error_function.evaluate(weights) + self.regularizer.evaluate(weights)
        if self.verbose:
            print error
        return error
        
    def _evaluate_gradient(self, weights):
        gradient = (self.error_function.evaluate_gradient(weights) + self.regularizer.evaluate_gradient(weights))/10
        return gradient
