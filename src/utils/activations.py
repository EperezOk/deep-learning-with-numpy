import numpy as np

def sigmoid(X: np.array):
    return 1 / (1 + np.exp(-X))

def step(X: np.array):
    return np.where(X >= 0, 1, -1)
