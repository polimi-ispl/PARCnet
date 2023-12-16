import numpy as np
import scipy.signal
import scipy.linalg
import warnings
from numba import njit


@njit
def apply_prediction_filter(past: np.ndarray, coeff: np.ndarray, steps: int) -> np.ndarray:
    prediction = np.zeros(steps)

    for i in range(steps):
        pred = np.dot(past, coeff)

        prediction[i] = pred

        past = np.roll(past, -1)
        past[-1] = pred

    return prediction


class ARModel:
    """
    AR model of order p.
    It finds the model parameters via the autocorrelation method and Levinson-Durbin recursion.
    It uses Numba jit complier to accelerate sample-by-sample inference.
    """

    def __init__(self, p: int, diagonal_load: float = 0.0):
        self.p = p
        self.diagonal_load = diagonal_load

        # Pre-compile Numba decorated function to expedite future calls
        apply_prediction_filter(past=np.zeros(self.p), coeff=np.ones(self.p), steps=1)

    def autocorrelation_method(self, valid: np.ndarray) -> np.ndarray:
        # Compute the sample autocorrelation function
        acf = scipy.signal.correlate(valid, valid, mode='full', method='auto')

        # Find the zeroth lag index
        zero_lag = len(acf) // 2

        # First column of the autocorrelation matrix
        c = acf[zero_lag:zero_lag + self.p]

        # Diagonal loading to improve conditioning
        c[0] += self.diagonal_load

        # Autocorrelation vector
        b = acf[zero_lag + 1:zero_lag + self.p + 1]

        # Solve the Toeplitz system of equations using the efficient Levinson-Durbin recursion
        ar_coeff = scipy.linalg.solve_toeplitz(c, b, check_finite=False)

        return ar_coeff

    def predict(self, valid: np.ndarray, steps: int) -> np.ndarray:
        # Find AR model parameters
        ar_coeff = self.autocorrelation_method(valid)

        # Apply linear prediction
        pred = apply_prediction_filter(
            past=valid[-self.p:],
            coeff=np.ascontiguousarray(ar_coeff[::-1], dtype=np.float32),  # needed for njit
            steps=steps
        )

        # Raise warning; helpful in case the AR model becomes numerically unstable.
        if np.any(np.abs(pred) > 1.0):
            warnings.warn(f'AR prediction exceeded the audio range [-1, 1]: found [{np.min(pred)}, {np.max(pred)}]',
                          RuntimeWarning)

        return pred
