import dataclasses

import numpy as np
from scipy.optimize import minimize


@dataclasses.dataclass
class OutbreakParams:
    loc: np.ndarray
    scale: np.ndarray
    pattern: np.ndarray




class LocScalePatternFinder:
    '''
    For each location, find the best yearly location, scale and pattern parameters for a given data set.
    so that y[l, s, m] = loc[l, s] + scale[l,s] * pattern[l,m] + noise
    '''

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def find_params(self) -> OutbreakParams:
        L, Y, M = self._data.shape

        # Initialize parameters with reasonable starting values
        loc_init = np.mean(self._data, axis=2)  # Mean across months for each location/year
        scale_init = np.ones((L, Y))
        pattern_init = np.zeros((L, M))

        # Initialize pattern as deviation from yearly means
        for loc_idx in range(L):
            centered = self._data[loc_idx] - loc_init[loc_idx][:, np.newaxis]
            pattern_init[loc_idx] = np.mean(centered, axis=0)

        # Optimize each location separately
        loc = np.zeros((L, Y))
        scale = np.zeros((L, Y))
        pattern = np.zeros((L, M))

        for loc_idx in range(L):
            # Define objective function for this location
            def objective(params: np.ndarray, idx: int = loc_idx) -> float:
                # Unpack parameters: [loc_0, loc_1, ..., loc_{Y-1}, scale_0, scale_1, ..., scale_{Y-1}, pattern_0, pattern_1, ..., pattern_{M-1}]
                loc_l = params[:Y]
                scale_l = params[Y:2*Y]
                pattern_l = params[2*Y:]

                # Compute predicted values
                predicted = loc_l[:, np.newaxis] + scale_l[:, np.newaxis] * pattern_l[np.newaxis, :]

                # Compute MSE
                mse: float = np.mean((self._data[idx] - predicted) ** 2)
                return mse

            # Initial parameter vector for this location
            x0 = np.concatenate([loc_init[loc_idx], scale_init[loc_idx], pattern_init[loc_idx]])

            # Optimize
            result = minimize(objective, x0, method='BFGS')

            # Extract optimized parameters
            loc[loc_idx] = result.x[:Y]
            scale[loc_idx] = result.x[Y:2*Y]
            pattern[loc_idx] = result.x[2*Y:]

        return OutbreakParams(loc=loc, scale=scale, pattern=pattern)

    def predict(self, params: OutbreakParams) -> np.ndarray:
        L, Y, M = self._data.shape
        predictions = np.zeros((L, Y, M))
        for loc_idx in range(L):
            predictions[loc_idx] = params.loc[loc_idx][:, np.newaxis] + params.scale[loc_idx][:, np.newaxis] * params.pattern[loc_idx][np.newaxis, :]
        return predictions
