import numpy as np

class LossFunction:
    """
    Base class for loss functions.
    """  
    def __call__(self, Y: np.array, P: np.array):
        """
        @param Y: target values
        @param P: predicted values
        """
        raise NotImplementedError
    
    def derivative(self, Y: np.array, P: np.array):
        """
        Derivative of the loss function, with respect to `P`.
        @param Y: target values
        @param P: predicted values
        """
        raise NotImplementedError

class MSE(LossFunction):
    """
    Mean Squared Error loss function.
    """  
    def __call__(self, Y: np.array, P: np.array):
        return np.mean(np.square(Y - P))
    
    def derivative(self, Y: np.array, P: np.array):
        return 2 * (P - Y)

mse = MSE()
