import numpy as np

def min_max_scale(
    X: np.ndarray,
    from_range: tuple[float, float],
    to_range = (0, 1),
) -> np.ndarray:
    """
    Scale the `X` array to the specified range.

    Parameters
    ----------
    X : numpy.ndarray
        Array to scale.
    from_range : tuple
        Range to scale `X` from, specified as a tuple (min, max).
    to_range : tuple
        Range to scale `X` to, specified as a tuple (min, max).

    Returns
    -------
    numpy.ndarray
        Scaled array.
    """
    X_std = (X - from_range[0]) / (from_range[1] - from_range[0])
    return X_std * (to_range[1] - to_range[0]) + to_range[0]
