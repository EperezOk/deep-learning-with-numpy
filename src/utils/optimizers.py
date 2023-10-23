import numpy as np

class Momentum:

    def __init__(self, shape, rate = 0.9):
        self.prev_dw = np.zeros(shape)
        self.rate = rate

    def __call__(self, dw: np.array):
        _dw = dw.copy()
        dw += self.rate * self.prev_dw
        self.prev_dw = _dw
        return dw
