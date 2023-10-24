import numpy as np

class ActivationFunction:
    """
    Base class for activation functions.
    """  
    def __call__(self, X: np.ndarray):
        """
        Take `X` and return the result of applying the activation function to it.
        """
        raise NotImplementedError

    def derivative(self, X: np.ndarray):
        """
        Take `X` and return the result of applying the derivative of the activation 
        function to it, w.r.t `X`.
        """
        raise NotImplementedError

class Step(ActivationFunction):
    """
    Step activation function.
    """  
    def __call__(self, X: np.ndarray):
        return np.where(X >= 0, 1, -1)

    def derivative(self, X: np.ndarray):
        # Technically not correct, but allows for generic gradient descent
        # to be used with the step function.
        return np.ones(X.shape)

step = Step()

class Linear(ActivationFunction):
    """
    Identity activation function.
    """  
    def __call__(self, X: np.ndarray):
        return X

    def derivative(self, X: np.ndarray):
        return np.ones(X.shape)

linear = Linear()

class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """  
    def __call__(self, X: np.ndarray):
        return 1 / (1 + np.exp(-X))

    def derivative(self, X: np.ndarray):
        _sigmoid = self.__call__(X)
        return _sigmoid * (1 - _sigmoid)

sigmoid = Sigmoid()
