import numpy as np

def get_linear_data():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Number of data points
    num_points = 50
    # Generate x values
    x = np.linspace(-5, 5, num_points)

    # Generate y values with a negative slope and some noise
    slope = -2
    intercept = 5
    noise = np.random.normal(0, 1, num_points)
    y = slope * x + intercept + noise

    return x.reshape(-1, 1), y


def get_quadratic_data():
    np.random.seed(42)

    num_points = 50

    x = np.linspace(-5, 5, num_points)

    intercept = 2
    noise = np.random.normal(0, 1, num_points)
    y = x**2 + intercept + noise

    return x.reshape(-1, 1), y


def get_exponential_data():
    np.random.seed(42)

    num_points = 50

    x = np.linspace(-5, 5, num_points)

    intercept = -2
    noise = np.random.normal(0, 2, num_points)
    y = np.exp(x) + intercept + noise

    return x.reshape(-1, 1), y
