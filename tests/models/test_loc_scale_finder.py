"""Tests for LocScalePatternFinder model."""
import numpy as np
import pytest

from chap_pymc.models.loc_scale_finder import LocScalePatternFinder


@pytest.fixture
def data() -> np.ndarray:
    n_loc, n_years, n_months = 2, 3, 4
    patterns = np.array([
        [0.0, 1.0, 2.0, 1.0],  # pattern for location 1
        [1.0, 2.0, 1.0, 0.0],  # pattern for location 2
    ])
    locs = np.array([
        [1.0, 2.0, 3.0],  # locs for location 1
        [2.0, 3.0, 4.0],  # loc
    ])
    scales = np.array([
        [1.0, 1.0, 1.0],  # scales for location 1
        [2.0, 2.0, 2.0],  # scales for location 2
    ])
    eta = patterns[:, np.newaxis, :]*scales[..., np.newaxis] + locs[..., np.newaxis]
    noise = np.random.rand(n_loc*n_years*n_months).reshape((n_loc, n_years, n_months))
    result: np.ndarray = eta + noise
    return result


def test_loc_scale_pattern_finder(data: np.ndarray) -> None:
    finder = LocScalePatternFinder(data)
    params = finder.find_params()
    predictions = finder.predict(params)
    print(predictions-data)
    assert params.scale.shape == (2, 3)
    assert params.loc.shape == (2, 3)
    assert params.pattern.shape == (2, 4)
