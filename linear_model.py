import numpy as np
from optimizer import BatchGradientDescent, LinearSolver, BFGS
from metrics import MSE, LogLoss
from regularizer import L2, L0
from utils import sigmoid, vector_binarize, add_bias_term, check_range

class LinearRegression:
    
    def __init__(self, solver = "SGD", learning_rate = 0.01, max_iter = 1500, verbose = False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.solver = solver
        
    def fit(self, x, y):
        X = add_bias_term(x)
        if self.solver == "SGD":
            mse = MSE(X, y, self.hypothesis)
            #print mse.evaluate(np.zeros(X.shape[1]))
            optimizer = BatchGradientDescent(dim = X.shape[1], learning_rate = self.learning_rate, error_function = mse, regularizer = L0())
            self.W = optimizer.minimize(max_iter = self.max_iter, verbose = self.verbose)
        else:
            optimizer = LinearSolver(X, y)
            self.W = optimizer.solve()
        
    def predict(self, x):
        if x.shape[1]+1 != self.W.shape[0]:
            print "Error: dimension mis-match"
            return
        X = add_bias_term(x)
        return self.hypothesis(self.W, X)
        
    def hypothesis(self, W, X):
        return X.dot(W)


        
class BinaryLogisticRegression:

    def __init__(self, solver = "BFGS", learning_rate = 0.001, C = 1.0, max_iter = 100000, verbose = False):
        self.learning_rate = learning_rate
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.solver = solver
        
    def fit(self, x, y):
        X = add_bias_term(x)
        self.classes = np.unique(y)
        logloss = LogLoss(X, y, self.hypothesis)
        if self.C > 0:
            regularizer = L2(X.shape[0], self.C)
        else:
            regularizer = L0()
        if self.solver == "SGD":
            optimizer = BatchGradientDescent(dim = X.shape[1], learning_rate = self.learning_rate, error_function = logloss, regularizer = regularizer)
            self.W = optimizer.minimize(max_iter = self.max_iter, verbose = self.verbose)
        else:
            optimizer = BFGS(dim = X.shape[1], error_function = logloss, regularizer = regularizer)
            self.W = optimizer.minimize(max_iter = self.max_iter, verbose = self.verbose)
        
    def predict_proba(self, x):
        if x.shape[1]+1 != self.W.shape[0]:
            print "Error: dimension mis-match"
            return
        X = add_bias_term(x)
        prob = np.zeros((X.shape[0], 2))
        prob[:,0] = self.hypothesis(self.W, X)
        prob[:,1] = 1 - prob[:,0]
        return prob
        
    def predict(self, x, class_boundry = 0.5):
        p = self.predict_proba(x)[:,0]
        p[np.where(p <= class_boundry )] = 0
        p[np.where(p > class_boundry )] = 1
        return p
        
    def hypothesis(self, W, X):
        return sigmoid(X.dot(W))
        
class LogisticRegression:

    def __init__(self, solver = "BFGS", learning_rate = 0.001, C = 1.0, max_iter = 100000, verbose = False):
        self.learning_rate = learning_rate
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.solver = solver
        self.models = []
        
    def fit(self, x, y):
        if(check_range(y) == False):
            print "Error: range must be between 0 and n_classes-1"
            return
        self.classes = np.unique(y)
        if self.classes.shape[0] < 2:
            print "Error: No. classes less than 2"
            return
        elif self.classes.shape[0] == 2:
            self.models.append(BinaryLogisticRegression(solver = self.solver, learning_rate = self.learning_rate, C = self.C, max_iter = self.max_iter, verbose = self.verbose))
            bin_y = vector_binarize(y, np.classes[1])
            self.models[0].fit(x, bin_y[:,1])
        else:
            bin_y = vector_binarize(y)
            for i in range(self.classes.shape[0]):
                self.models.append(BinaryLogisticRegression(solver = self.solver, learning_rate = self.learning_rate, C = self.C, max_iter = self.max_iter, verbose = self.verbose))
                self.models[i].fit(x, bin_y[:,i])
        
    def predict_proba(self, x):
        prob = np.zeros((x.shape[0], self.classes.shape[0]))
        if self.classes.shape[0] == 2:
            prob = self.models[i].predict_proba(x)
        else:
            for i in range(self.classes.shape[0]):
                #print self.models[i].predict_proba(x).shape, prob.shape
                prob[:,i] = self.models[i].predict_proba(x)[:,0]
        return prob
        
    def predict(self, x):
        prob = self.predict_proba(x)
        return np.argmax(prob, axis=1)
