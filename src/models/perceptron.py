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
        self.weight_history = []
        self.predic_history = []


    def fit(
        self,
        inputs: np.array,
        outputs: np.array,
        epochs = 300,
        tolerance=1e-3,
        save_w_history = False,
        save_p_history = False
    ):
        """
        Trains the perceptron to fit the `inputs` to the `outputs`.
        @param inputs: (N, M)
        @param outputs: (N, 1)
        """
        _inputs = np.insert(inputs, 0, 1, axis=1) # bias at the start of each input

        for epoch in range(epochs):
            predictions = self.predict(inputs)

            loss = mse(outputs, predictions)
            if (loss <= tolerance): break # early stopping

            errors = outputs - predictions
            errors = errors.reshape(-1, 1) # reshape to multiply each row as a scalar w/inputs

            dw = self.lr * errors * _inputs # (N, M+1)
            dw = self.optimizer.apply(dw)
            dw = np.sum(dw, axis=0) # (1, M+1), squash deltas

            self.weights += dw

            if (epoch % 5 == 0 and save_w_history): self.weight_history.append(self.weights.copy())
            if (epoch % 2 == 0 and save_p_history): self.predic_history.append(predictions.copy())
            if (epoch % 10 == 0): print(f"{epoch=} ; {loss=}")
        
        self.weight_history.append(self.weights.copy())
        self.predic_history.append(predictions.copy())
        print(f"{epoch=} ; {loss=}")


    def predict(self, inputs: np.array):
        """
        Makes a prediction over `inputs`.
        @param inputs: (N, M)
        @return predictions: (N, 1)
        """
        _inputs = np.insert(inputs, 0, 1, axis=1) # bias at the start of each input
        H = _inputs @ self.weights # (N, 1)
        return self.activation(H)
