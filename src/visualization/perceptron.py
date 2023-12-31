import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation

def plot_weight_history(inputs, labels, weight_history, xlim=[0, 6], ylim=[0, 7]):
    """
    Plots the update of the weights over the epochs.
    """
    fig, ax = plt.subplots()

    def update(i):
        ax.clear()

        weights = weight_history[i]

        # Plot the training data
        plt.scatter(
            x=inputs[:, 0],
            y=inputs[:, 1],
            color=np.where(labels == 1, "r", "b"),
        )

        x = np.linspace(xlim[0], xlim[1], 100)

        # w0 + w1*x + w2*y = 0 => y = -(w1*x + w0) / w2
        y = -(weights[1] * x + weights[0]) / weights[2]

        # Plot the separating hyperplane
        ax.plot(x, y, c="g")

        legend_elements = [
            Patch(color="blue", label="Group A"),
            Patch(color="red", label="Group B"),
            Line2D([0], [0], color="green", label="Separating hyperplane")
        ]

        ax.legend(handles=legend_elements, loc="lower right")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Epoch {i*5}")

    anim = FuncAnimation(fig, update, frames=len(weight_history), interval=500)

    anim.save("out/weight_history.gif")
    fig.savefig("out/separating_hyperplane.png") # save the last frame

    fig.clf()


def plot_predict_history(inputs, predic_history, outputs, ylim=[-10, 20], interval=1):
    """
    Plots the predictions for the `inputs` over the epochs.
    """
    fig, ax = plt.subplots()

    def update(i):
        ax.clear()

        predictions = predic_history[i]

        # Plot the expected outputs
        plt.scatter(inputs, outputs, label="Data")

        # Plot the predictions
        plt.plot(inputs, predictions, "r", marker="o", markersize=3, label="Predictions")

        ax.set_title(f"Epoch {i*interval}")
        ax.legend()
        ax.set_ylim(ylim)

    anim = FuncAnimation(fig, update, frames=len(predic_history), interval=300)

    anim.save("out/predict_history.gif")
    fig.savefig("out/regression.png") # save the last frame

    fig.clf()
