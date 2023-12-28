import numpy as np

from utils.activations import relu, ActivationFunction
from utils.losses import mse
from utils.optimizers import Momentum

class MLP():

    def __init__(
        self, lr: float, layers: list[int], activation_out: ActivationFunction
    ):
        """
        Receives a list of layer sizes, including the input layer size and the output layer size.

        The activation function for all hidden layers is ReLU.
        """
        self.learning_rate = lr

        # +1 in the column dim to account for bias term on each layer
        self.weights = [np.random.randn(layers[i + 1], layers[i] + 1) for i in range(len(layers) - 1)]

        self.activation_out = activation_out

        # momentum optimization
        self.optimizers = [Momentum(weights.shape) for weights in self.weights]


    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 10_000):
        for epoch in range(epochs):
            H, V, hO, O = self.feed_forward(X)

            loss = mse(Y, O)
            loss_derivative = mse.derivative(Y, O)

            if epoch % 1000 == 0: print(f"{epoch=} ; {loss=}")

            self.backpropagation(H, V, hO, loss_derivative)

        return O


    def feed_forward(self, inputs: np.ndarray):
        # add neuron with constant output 1 to inputs, to account for bias
        X = np.insert(inputs, 0, 1, axis=1)

        V = [X.T]
        H = []

        # iterate over hidden layers
        for i, w in enumerate(self.weights[:-1]):
            h = w @ V[i]
            v = relu(h)
            v = np.insert(v, 0, 1, axis=0) # add bias to hidden layer output
            H.append(h)
            V.append(v)

        hO = self.weights[-1] @ V[-1]

        O = self.activation_out(hO)

        # transpose output `O` to shape (N, output_dim)
        return H, V, hO, O.T


    def backpropagation(
        self,
        H: np.ndarray,
        V: np.ndarray,
        hO: np.ndarray,
        loss_derivative: np.ndarray
    ):
        # update output layer weights
        deltas = loss_derivative.T * self.activation_out.derivative(hO)
        dw = self.learning_rate * deltas @ V[-1].T

        dW = [dw]

        # iterate backwards over hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            # remove bias from output layer weights
            prev_delta_sum = deltas.T @ self.weights[i+1][:, 1:]

            deltas = prev_delta_sum.T * relu.derivative(H[i])
            # V[i] is the output of the (i-1)-th hidden layer, since V[0] is the input (and len(V) = len(H) + 1)
            # insert at the beginning to keep the order
            dW.insert(0, -self.learning_rate * deltas @ V[i].T)

        # final gradient for input layer
        prev_delta_sum = deltas.T @ self.weights[0][:, 1:]

        # update weights, using momentum optimization
        for i in range(len(self.weights)):
            self.weights[i] += self.optimizers[i](dW[i])

        return prev_delta_sum
