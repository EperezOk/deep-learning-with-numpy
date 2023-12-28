import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models.mlp import MLP
from utils.activations import sigmoid

def main():
    X = parse_digits("src/data/digits/digits.txt")
    # expected_output for 0 is [1 0 0 .... 0], for 1 is [0 1 0 .... 0], for 9 is [0 0 .... 0 1]
    y = np.eye(10)

    mlp = MLP(lr=0.0005, layers=[X.shape[1], 10, 10], activation_out=sigmoid)

    # train the model
    mlp.fit(X, y)

    # predict a digit from the training set
    clear_digit = X[5].reshape(1, -1)
    _,_,_, prediction = mlp.feed_forward(clear_digit)

    print(f"\n{prediction}")
    print(f"\nPrediction is {np.argmax(prediction)}\n")
    visualize_digit(clear_digit)

    # predict noisy digits
    for i in [2,3,8]:
        noisy_digit = parse_digits(f"src/data/digits/{i}_with_noise.txt")
        _,_,_, prediction = mlp.feed_forward(noisy_digit)

        print(prediction)
        print(f"\nPrediction is {np.argmax(prediction)}\n")
        visualize_digit(noisy_digit)


def parse_digits(path: str):
    with open(path, 'r') as f:
        data = f.read().splitlines()

    digit_images = []
    
    for i, line in enumerate(data):
        if i % 7 == 0:
            digit_img = []
        numbers = line.strip().split()
        digit_img.extend(numbers)
        if i % 7 == 6:
            digit_images.append(digit_img)

    return np.array(digit_images, dtype=float)


def visualize_digit(digit: np.array):
	sns.heatmap(digit.reshape(7, 5), cmap='Greys', vmin=0, vmax=1)
	plt.show()


if __name__ == "__main__":
    main()
