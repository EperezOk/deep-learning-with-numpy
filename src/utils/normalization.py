import numpy as np

def min_max_scale(
    X: np.array,
    from_range: tuple[float, float],
    to_range = (0, 1),
) -> np.array:
    """
    Scales the `X` array to the range `to_range`.
    @param X: array to scale.
    @param to_range: range to scale to.
    @return: scaled array.
    """
    X_std = (X - from_range[0]) / (from_range[1] - from_range[0])
    return X_std * (to_range[1] - to_range[0]) + to_range[0]
