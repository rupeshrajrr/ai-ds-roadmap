import pytest
import numpy as np
from day2_numpy.numpy_exercises import min_max_scale, standardise, moving_average_1d, one_hot_encode, softmax

# 12 Assertions (4 test matrices x 3 properties checked)
@pytest.mark.parametrize("mock_data", [
    np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    np.array([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]]),  # Zero-variance column
    np.ones((4, 4)) * 7.0,                           # Flat matrix
    np.random.RandomState(42).randn(10, 3)           # Static random distribution
])
def test_scalers(mock_data):
    scaled = min_max_scale(mock_data)
    standardised = standardise(mock_data)
    
    assert scaled.shape == mock_data.shape
    assert standardised.shape == mock_data.shape
    assert np.all(scaled >= -1e-9) and np.all(scaled <= 1.0 + 1e-9)

# 10 Assertions (5 distinct configurations evaluated comprehensively)
@pytest.mark.parametrize("arr, w, expected_len, sample_idx, expected_val", [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3, 3, 0, 2.0),
    (np.arange(10), 5, 6, 2, 4.0),
    (np.ones(100), 10, 91, 50, 1.0),
    (np.array([10.0, 20.0, 30.0, 40.0]), 2, 3, 1, 25.0),
    (np.array([1.5, 2.5, 3.5, 4.5, 5.5]), 4, 2, 0, 3.0)
])
def test_moving_average(arr, w, expected_len, sample_idx, expected_val):
    res = moving_average_1d(arr, w)
    assert len(res) == expected_len
    assert np.isclose(res[sample_idx], expected_val)

# 8 Assertions (4 parameter variations x 2 structural checks)
@pytest.mark.parametrize("labels, classes", [
    (np.array([0, 1, 2]), 3),
    (np.array([0, 4, 2, 1]), 5),
    (np.array([0, 0, 0]), 1),
    (np.array([3, 2, 1, 0]), 4)
])
def test_one_hot(labels, classes):
    encoded = one_hot_encode(labels, classes)
    assert encoded.shape == (len(labels), classes)
    assert np.all(np.sum(encoded, axis=1) == 1)

# 7 Assertions (All check mathematical convergence properties across matrices)
@pytest.mark.parametrize("X", [
    np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]), # Stability test
    np.array([[0.0, 0.0], [-1.0, -2.0]]),
    np.random.RandomState(42).randn(5, 5)
])
def test_softmax_properties(X):
    prob = softmax(X)
    assert prob.shape == X.shape
    assert np.allclose(np.sum(prob, axis=1), 1.0)
    assert np.all(prob >= 0.0)
