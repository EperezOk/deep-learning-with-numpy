import numpy as np

def step(X: np.array):
    return np.where(X >= 0, 1, -1)

def linear(X: np.array):
    return X

def sigmoid(X: np.array):
    return 1 / (1 + np.exp(-X))
