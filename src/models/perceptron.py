import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
        self.history = []
        self.optimizer = Momentum(self.weights.shape)


    def fit(self, inputs: np.array, outputs: np.array, epochs = 1000):
        """
        Trains the perceptron to fit the `inputs` to the `outputs`.
        @param inputs: (N, M)
        @param outputs: (N, 1)
        """
        _inputs = np.insert(inputs, 0, 1, axis=1) # bias at the start of each input

        for epoch in range(epochs):
            predictions = self.predict(inputs)

            loss = mse(outputs, predictions)
            if (loss == 0): break # early stopping

            errors = outputs - predictions
            errors = errors.reshape(-1, 1) # reshape to multiply each row as a scalar w/inputs

            dw = self.lr * errors * _inputs # (N, M+1)
            dw = self.optimizer.apply(dw)
            dw = np.sum(dw, axis=0) # (1, M+1), squash deltas

            self.weights += dw

            if (epoch % 5 == 0): self.history.append(self.weights.copy())
            if (epoch % 10 == 0): print(f"{epoch=} ; {loss=}")
        
        self.history.append(self.weights.copy())
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
    

    def plot_hyperplane(self, inputs = [], labels = []):
        colors = np.array(["blue" if o == -1 else "red" for o in labels])

        plt.scatter(inputs[:, 0], inputs[:, 1], color=colors)

        x = np.linspace(0, 6, 20)
        hyperplane = -(self.weights[0] + self.weights[1] * x) / self.weights[2]

        plt.plot(x, hyperplane, "g")

        plt.xlim(0, 6)
        plt.ylim(0, 6)
        plt.grid()
        plt.savefig("out/step_perceptron.png")
        plt.close()


    def plot_history(self, inputs = [], labels = []):
        """
        Plots the update of the weights over the epochs.
        """
        fig, ax = plt.subplots()

        def update(i):
            ax.clear()

            weights = self.history[i]

            # Plot the training data
            plt.scatter(
                x=inputs[:, 0],
                y=inputs[:, 1],
                color=np.where(labels == 1, "r", "b"),
            )

            xmax, ymax = np.max(inputs[:, 0]), np.max(inputs[:, 1])
            x = np.linspace(0, xmax + 10, 100)

            # w0 + w1*x + w2*y = 0 => y = -(w1*x + w0) / w2
            y = -(weights[1] * x + weights[0]) / weights[2]

            # Plot the separating hyperplane
            ax.plot(x, y, c="g")

            ax.set_xlim([0, xmax+2])
            ax.set_ylim([0, ymax+2])
            ax.set_title(f"Epoch {i*5}")
            ax.grid()

        anim = FuncAnimation(fig, update, frames=len(self.history), interval=500)

        anim.save("out/step_perceptron.gif")
        fig.clf()
