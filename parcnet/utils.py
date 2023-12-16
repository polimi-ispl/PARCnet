import numpy as np
from copy import deepcopy


def simulate_packet_loss(y_ref: np.ndarray, trace: np.ndarray, packet_dim: int) -> np.ndarray:
    # Copy the clean signal to create the lossy signal
    y_lost = deepcopy(y_ref)

    # Simulate packet losses according to given trace
    for i, loss in enumerate(trace):
        if loss:
            idx = i * packet_dim
            y_lost[idx: idx + packet_dim] = 0.

    return y_lost
