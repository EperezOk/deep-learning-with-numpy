from data.linear_classification import inputs as i1, outputs as o1
from utils.activations import step
from models.perceptron import Perceptron

def main():
    step_perceptron()


def step_perceptron():
    perceptron = Perceptron(M=2, activation=step)

    perceptron.fit(i1, o1, epochs=300)

    perceptron.plot_history(i1, o1)
    perceptron.plot_hyperplane(i1, o1)


if __name__ == "__main__":
    main()
