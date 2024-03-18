import mlx.core as mx
import numpy as np


def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    """
    From DeepCache, calculate a sampling frequency distribution that bias towards less similar features.

    :param total_numbers: Total numbers of de-noising steps.
    :param n_samples: Number
    """
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace(
            (-center)**(1/pow), (total_numbers-center)**(1/pow), n_samples+1)
        indices = [0] + \
            [x+center for x in np.unique(np.int32(x_values**pow))[1:-1]]
        if len(indices) == n_samples:
            break
        pow -= 0.02
    if pow <= 1:
        raise ValueError(
            "Cannot find suitable pow. Please adjust n_samples or decrease center.")
    return indices, pow
