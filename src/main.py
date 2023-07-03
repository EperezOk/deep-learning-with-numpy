from data import regression, classification
from utils.activations import step, linear
from models.perceptron import Perceptron
import visualization.perceptron as p_viz

def main():
    # step_perceptron()
    linear_perceptron()


def step_perceptron():
    inputs, outputs = classification.get_linear_data()

    perceptron = Perceptron(M=2, activation=step)

    perceptron.fit(inputs, outputs, tolerance=0, save_w_history=True)

    p_viz.plot_weight_history(inputs, outputs, perceptron.weight_history)
    p_viz.plot_hyperplane(inputs, outputs, perceptron.weights)

    # Show how XOR is not linearly separable


def linear_perceptron():
    inputs, outputs = regression.get_linear_data()

    perceptron = Perceptron(M=1, lr=0.001, activation=linear)

    perceptron.fit(inputs, outputs, tolerance=0.83, save_p_history=True)

    p_viz.plot_predic_history(
        inputs.reshape(1,-1)[0],
        [p.reshape(1,-1)[0] for p in perceptron.predic_history],
        outputs,
    )


if __name__ == "__main__":
    main()
