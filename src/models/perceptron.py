import numpy as np

from utils.optimizers import Momentum
from utils.activations import sigmoid
from utils.losses import mse

class Perceptron:

    def __init__(self, M: int, lr = 0.01, activation = sigmoid):
        """
        @param M: amount of features of each input.
        @param lr: learning rate.
        @param activation: activation function.
        """
        self.lr = lr
        self.activation = activation
        self.weights = np.zeros(M + 1) # + 1 for bias
        self.optimizer = Momentum(self.weights.shape)


    def fit(
        self,
        inputs: np.array,
        outputs: np.array,
        epochs = 300,
        tolerance = 1e-3,
        threshold_predictions = lambda P: P,
    ) -> tuple[list[np.array], list[np.array]]:
        """
        Trains the perceptron to fit the `inputs` to the `outputs`.
        @param inputs: (N, M)
        @param outputs: (N, 1)
        @param epochs: max amount of epochs to train.
        @param tolerance: min loss to stop training.
        @param threshold_predictions: callable that thresholds the predictions.
        @return1 weight_history: list of weights for each epoch.
        @return2 predict_history: list of predictions for each epoch.
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
            dw = self.optimizer.apply(dw)
            dw = np.sum(dw, axis=0) # (1, M+1), squash deltas

            self.weights += dw

            if (epoch % 5 == 0): weight_history.append(self.weights.copy())
            predict_history.append(predictions.copy())

            if (epoch % 10 == 0): print(f"{epoch=} ; {loss=}")
        
        weight_history.append(self.weights.copy())
        predict_history.append(predictions.copy())
        print(f"{epoch=} ; {loss=}")

        return weight_history, predict_history


    def predict(self, inputs: np.array) -> tuple[np.array, np.array]:
        """
        Makes a prediction over `inputs`.
        @param inputs: (N, M)
        @return1 predictions: (N, 1)
        @return2 H: (N, 1), linear combination of the inputs and the weights
        """
        _inputs = np.insert(inputs, 0, 1, axis=1) # bias at the start of each input
        H = _inputs @ self.weights # (N, 1)
        return self.activation(H), H
