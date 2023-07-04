import numpy as np

from data import regression, classification
from utils.normalization import min_max_scale
from utils.activations import step, linear, sigmoid
from models.perceptron import Perceptron
import visualization.perceptron as p_viz

def main():
    step_perceptron()
    linear_perceptron()
    non_linear_perceptron()


def step_perceptron():
    inputs, outputs = classification.get_linear_data()

    perceptron = Perceptron(M=2, activation=step)

    weight_history, _ = perceptron.fit(inputs, outputs, tolerance=0)

    p_viz.plot_weight_history(inputs, outputs, weight_history)

    # Show how XOR is not linearly separable


def linear_perceptron():
    inputs, outputs = regression.get_linear_data()

    perceptron = Perceptron(M=1, lr=0.001, activation=linear)

    _, predict_history = perceptron.fit(inputs, outputs, tolerance=0.83)

    p_viz.plot_predict_history(
        inputs.reshape(1,-1)[0],
        [p.reshape(1,-1)[0] for p in predict_history],
        outputs,
    )


def non_linear_perceptron():
    # --------- Classification ----------

    inputs, outputs = classification.get_linear_data()

    # Normalization to [0,1] is not necessary, but it helps the perceptron converge faster
    inputs = min_max_scale(inputs, (inputs.min(), inputs.max()))

    perceptron = Perceptron(M=2, activation=sigmoid)

    weight_history, _ = perceptron.fit(
        inputs,
        outputs,
        tolerance=0,
        threshold_predictions=lambda P: np.where(P > 0.5, 1, -1),
    )

    # The plot will be scaled down to the range of the inputs
    p_viz.plot_weight_history(
        inputs,
        outputs,
        weight_history,
        xlim=[-0.5, 1.5],
        ylim=[-1.5, 2]
    )

    # -------- Regression ---------

    # inputs, outputs = regression.get_quadratic_data() # performs badly
    inputs, outputs = regression.get_exponential_data() # performs well
    # inputs, outputs = regression.get_linear_data()    # performs well

    # In regression, normalization is necessary for the outputs
    # inputs = min_max_scale(inputs) # performs worse
    original_range = (outputs.min(), outputs.max())
    _outputs = min_max_scale(outputs, from_range=original_range)

    perceptron = Perceptron(M=1, lr=0.06, activation=sigmoid)

    _, predict_history = perceptron.fit(inputs, _outputs, epochs=400)

    predict_history = [p.reshape(1,-1)[0] for p in predict_history[::5]]

    p_viz.plot_predict_history(
        inputs.reshape(1,-1)[0],
        # scale predictions back to the original output range
        [min_max_scale(p, from_range=(0,1), to_range=original_range) for p in predict_history],
        outputs,
        ylim=[-10, 200],
        interval=5, # show every 5 epochs
    )


if __name__ == "__main__":
    main()
