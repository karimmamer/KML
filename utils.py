import numpy as np

def sigmoid(M):
    return 1.0 / (1.0 + np.exp(-1.0*M))

def sigmoid_gradient(M):
    return sigmoid(M) * (1-sigmoid(M))
    
def vector_binarize(vec):
    unique = np.unique(vec)
    vec_bin = np.zeros((vec.shape[0], unique.shape[0]))
    for u in unique:
        vec_bin[np.where(vec == u), u] = 1
    return vec_bin
    
def add_bias_term(x):
    X = np.ones((x.shape[0], x.shape[1]+1))
    X[:,1:] = x
    return X
    
def check_range(y):
    u = np.unique(y)
    for i in range(u.shape[0]):
        if i != u[i]:
            return False
    return True
    
def zero_initializer(shape):
    return np.zeros(shape)
    
def random_initializer(shape, low, high):
    return np.random.uniform(low = low, high = high, size = shape)
