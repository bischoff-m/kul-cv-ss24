# %%
import numpy as np


def triangle_wave(x: float, period: float) -> float:
    """Triangle wave function in the range [0, 1].

    Args:
        x (float): Phase of the wave.
        period (float): Period of the wave.

    Returns:
        float: Value of the wave at x.
    """
    return 2 * np.abs(saw_wave(x + period / 2, period) - 0.5)


def saw_wave(x: float, period: float) -> float:
    """Saw wave function in the range [0, 1].

    Args:
        x (float): Phase of the wave.
        period (float): Period of the wave.

    Returns:
        float: Value of the wave at x.
    """
    return x / period - np.floor(x / period)


def rect(x: float, start: float, width: float) -> float:
    """Rectangular function.

    Args:
        x (float): Input value.
        start (float): Start of the rectangle.
        width (float): Width of the rectangle.

    Returns:
        float: 1 if x is inside the rectangle, 0 otherwise.
    """
    return 1 if start <= x < start + width else 0


def step(x: float, start: float) -> float:
    """Step function.

    Args:
        x (float): Input value.
        start (float): Start of the step.

    Returns:
        float: 1 if x is greater than or equal to start, 0 otherwise.
    """
    return 1 if x >= start else 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 3, 1000)
    y = saw_wave(x, 1)
    plt.plot(x, y)
    plt.show()
