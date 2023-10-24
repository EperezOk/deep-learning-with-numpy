import numpy as np

class LossFunction:
    """
    Base class for loss functions.
    """
    
    def __call__(self, Y, P) -> float:
        """
        Calculate the mean loss between target values and predicted values.

        Parameters
        ----------
        Y : np.ndarray
            Target values.
        P : np.ndarray
            Predicted values.

        Returns
        -------
        float
            The calculated mean loss.
        """
        raise NotImplementedError
    
    def derivative(self, Y, P) -> np.ndarray:
        """
        Calculate the derivative of the loss function with respect to predicted values.

        Parameters
        ----------
        Y : np.ndarray
            Target values.
        P : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            The calculated derivative of the loss.
        """
        raise NotImplementedError

class MSE(LossFunction):
    """
    Mean Squared Error loss function.
    """  
    def __call__(self, Y: np.ndarray, P: np.ndarray):
        L = np.square(Y - P)
        return np.mean(L)
    
    def derivative(self, Y: np.ndarray, P: np.ndarray):
        return 2 * (P - Y)

mse = MSE()
