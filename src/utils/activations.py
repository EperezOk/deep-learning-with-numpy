import numpy as np

class ActivationFunction:
    """
    Base class for activation functions.
    """  
    def __call__(self, X: np.array):
        """
        @param X: input values
        """
        raise NotImplementedError

    def derivative(self, X: np.array):
        # Derivative of the activation function, with respect to `X`.
        """
        @param X: input values
        """
        raise NotImplementedError

class Step(ActivationFunction):
    """
    Step activation function.
    """  
    def __call__(self, X: np.array):
        return np.where(X >= 0, 1, -1)

    def derivative(self, X: np.array):
        # Technically not correct, but allows for generic gradient descent
        # to be used with the step function.
        return np.ones(X.shape)

step = Step()

class Linear(ActivationFunction):
    """
    Identity activation function.
    """  
    def __call__(self, X: np.array):
        return X

    def derivative(self, X: np.array):
        return np.ones(X.shape)

linear = Linear()

class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """  
    def __call__(self, X: np.array):
        return 1 / (1 + np.exp(-X))

    def derivative(self, X: np.array):
        _sigmoid = self.__call__(X)
        return _sigmoid * (1 - _sigmoid)

sigmoid = Sigmoid()
