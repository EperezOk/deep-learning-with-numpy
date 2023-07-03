import numpy as np

def mse(Y: np.array, P: np.array):
    """
    @param Y: target values
    @param P: predicted values
    """
    return np.mean(np.square(Y - P))
