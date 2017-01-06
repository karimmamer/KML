import numpy as np
from optimizer import BatchGradientDescent, LinearSolver
from scipy.optimize import fmin_cg
from metrics import MSE, LogLoss
from regularizer import L2, L0
from utils import sigmoid, vector_binarize, add_bias_term, check_range
from utils import sigmoid_gradient, zero_initializer, random_initializer

class FeedForwardNNClassifier:

    def __init__(self, architecture, C = 1.0, solver = "CG", learning_rate = 0.01, max_iter = 200, verbose = False, intialize = "random"):
        self.architecture = architecture
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.intialize = intialize
        self.C = C
        self.weights = []
        self.activations = []
        self.activations_gradient = []
    
    def _set_architecture(self):
        last_dim = self.x.shape[1] + 1
        for layer in self.architecture:
            if layer['activation'] == 'sigmoid':
                self.activations.append(sigmoid)
                self.activations_gradient.append(sigmoid_gradient)
            if self.intialize == "random":
                self.weights.append(random_initializer((layer['size'], last_dim), -0.12, 0.12))
            else:
                self.weights.append(zero_initializer((layer['size'], last_dim)))
            last_dim = layer['size']+1
        
    def fit(self, x, y):
        self.x = x
        self.y = vector_binarize(y)
        self._set_architecture()
        if self.solver == "CG":
            self.W = self._weights_unrolling(fmin_cg(self._evaluate_error, self._weights_rolling(self.weights), fprime = self._evaluate_gradient, maxiter = self.max_iter, disp = True))
        
    def hypothesis(self, x, weights):
        output = []
        a = x
        for i in range(len(self.activations)):
            z = np.dot(add_bias_term(a), weights[i].T)
            output.append((z, self.activations[i](z)))
            a = output[i][1]
        return output
        
    def predict_proba(self, x):
        #prob = np.zeros((x.shape[0], 2))
        prob = self.hypothesis(x, self.W)[-1][1]
        #prob[:,1] = 1 - prob[:,0]
        return prob
        
    def predict(self, x):
        prob = self.predict_proba(x)
        return np.argmax(prob, axis=1)
        
    def _evaluate_error(self, weights_vector):
        weights = self._weights_unrolling(weights_vector)
        h = self.hypothesis(self.x, weights)[-1][1]
        
        error = np.sum(-1.0 * self.y * np.log(h) - (1.0 - self.y) * np.log(1.0 - h)) / self.x.shape[0]
        
        regularized_error = 0
        for w in weights:
            regularized_error += self.C * np.sum(w[:, 1:]**2)/ (2.0 * self.x.shape[0])
        
        print error + regularized_error
        
        return error + regularized_error
        
    def _evaluate_gradient(self, weights_vector):
        weights = self._weights_unrolling(weights_vector)
        h = self.hypothesis(self.x, weights)
        gradient = []
        
        for i in reversed(range(len(self.activations_gradient))):
        
            if i == len(self.activations_gradient) - 1:
                last_delta = h[i][1] - self.y
                continue
        
            current_delta = np.dot(last_delta, weights[i+1])[:, 1:]
            z = h[i][0]
            #z = np.hstack((np.zeros((h[i][0].shape[0], 1)), h[i][0]))
            #print z.shape, current_delta.shape
            current_delta *= self.activations_gradient[i](z)
            
            gradient.append(np.dot(last_delta.T, add_bias_term(h[i][1])) / self.x.shape[1])
            last_delta = current_delta
        
        gradient.append(np.dot(last_delta.T, add_bias_term(self.x)) / self.x.shape[1])

        gradient.reverse()
        
        for i in range(len(gradient)):
            #print weights[i].shape, gradient[i].shape
            weight_reg_gradient = np.zeros(weights[i].shape)
            weight_reg_gradient[:, 1:] = weights[i][:, 1:]
            weight_reg_gradient *= 1.0 * self.C / self.x.shape[1]
            gradient[i] += weight_reg_gradient
        
        return self._weights_rolling(gradient)
        
    def _weights_rolling(self, weights):
        size = 0
        for w in weights:
            size += w.shape[0] * w.shape[1]
        weights_vector = np.zeros((size,))
        begin = 0
        for w in weights:
            end = begin + w.shape[0] * w.shape[1]
            weights_vector[begin: end] = w.flatten()
            begin = end
        return weights_vector
        
    def _weights_unrolling(self, weights_vector):
        weights = []
        last_dim = self.x.shape[1] + 1
        begin = 0
        for layer in self.architecture:
            weights.append(np.zeros((layer['size'], last_dim)))
            end = begin + layer['size'] * last_dim
            weights[-1] = weights_vector[begin: end].reshape((layer['size'], last_dim))
            begin = end
            last_dim = layer['size'] + 1
        return weights
