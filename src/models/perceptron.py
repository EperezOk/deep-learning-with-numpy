import numpy as np

from utils.optimizers import Momentum
from utils.activations import sigmoid
from utils.losses import mse

class Perceptron:

    def __init__(self, M: int, lr = 0.01, activation = sigmoid):
        """
        Parameters
        ----------
        M : int
            Amount of features of each input.
        lr : float
            Learning rate.
        activation : str
            Activation function.
        """
        self.lr = lr
        self.activation = activation
        self.weights = np.zeros(M + 1) # + 1 for bias
        self.optimizer = Momentum(self.weights.shape)


    def fit(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        epochs = 300,
        tolerance = 1e-3,
        threshold_predictions = lambda P: P,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Trains the perceptron to fit the `inputs` to the `outputs`.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input data of shape (N, M).
        outputs : numpy.ndarray
            Target outputs of shape (N, 1).
        epochs : int
            Maximum number of epochs to train.
        tolerance : float
            Minimum loss value to stop training.
        threshold_predictions : callable
            A function that thresholds the predictions.

        Returns
        -------
        weight_history : list
            A list of weights for each epoch.
        predict_history : list
            A list of predictions for each epoch.
        """
        _inputs = np.insert(inputs, 0, 1, axis=1) # bias at the start of each input

        weight_history = []
        predict_history = []

        for epoch in range(epochs):
            predictions, H = self.predict(inputs)
            predictions = threshold_predictions(predictions)

            loss = mse(outputs, predictions)
            if (loss <= tolerance): break # early stopping

            # Works for step activation as well, thanks to how we defined these derivatives
            gradients = mse.derivative(outputs, predictions) * self.activation.derivative(H)
            gradients = gradients.reshape(-1, 1) # reshape to multiply each row as a scalar w/inputs

            dw = -1 * self.lr * gradients * _inputs # (N, M+1)
            dw = self.optimizer(dw)
            dw = np.sum(dw, axis=0) # (1, M+1), squash deltas

            self.weights += dw

            if (epoch % 5 == 0): weight_history.append(self.weights.copy())
            predict_history.append(predictions.copy())

            if (epoch % 10 == 0): print(f"{epoch=} ; {loss=}")
        
        weight_history.append(self.weights.copy())
        predict_history.append(predictions.copy())
        print(f"{epoch=} ; {loss=}")

        return weight_history, predict_history


    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction over `inputs`.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input data of shape (N, M).

        Returns
        -------
        predictions : numpy.ndarray
            Predicted outputs of shape (N, 1).
        H : numpy.ndarray
            Linear combination of the inputs and the weights, with shape (N, 1).
        """
        _inputs = np.insert(inputs, 0, 1, axis=1) # bias at the start of each input
        H = _inputs @ self.weights # (N, 1)
        return self.activation(H), H
