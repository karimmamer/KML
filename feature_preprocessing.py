import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = X.mean(0)
        self.std = X.std(0)

    def transform(self, X):
        if self.mean is None:
            print "call fit firstly"
            return
            
        if X.shape[1] != self.mean.shape[0]:
            print "dimension mis-match"
            return
        X = X.astype(np.float64)
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-self.mean[i])/self.std[i]
            
        return X
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        

class PCA:

    def __init__(self, method = 'pca'):
        self.method = method
        
    def fit(self, X):
        if self.method == 'svd':
            y = X
            self.U, self.s, self.V = np.linalg.svd(y, full_matrices = True)
            #self.V = self.V.T
            self.s = self.s**2
            #self.s = np.diag(s_arr)
        else:
            n = X.shape[0]
            X = X.T
            c = X.dot(X.T) / (n - 1)
            self.s, self.V = np.linalg.eig(c)
            #print self.s2.shape, self.V.shape 
            idx = self.s.argsort()[::-1]   
            self.s = self.s[idx]
            self.V = self.V[:,idx]
            self.V = self.V.T
            #print self.s2
        
    def transform(self, X, k):
        X = X.T
        return self.V[:k, :].dot(X).T
        
    def fit_transform(self, X, k):
        self.fit(X)
        return self.transform(X, k)
