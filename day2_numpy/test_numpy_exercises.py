"""
tests/test_numpy_exercises.py
─────────────────────────────
pytest tests for numpy_exercises.py — Day 2 (37 tests, all must pass)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from numpy_exercises import (
    broadcast_add,
    normalize_columns,
    pairwise_cosine_similarity,
    row_col_mean_diff,
    sliding_window_sum,
)

class TestBroadcastAdd:
    def test_same_shape_1d(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert_allclose(broadcast_add(a, b), np.array([5.0, 7.0, 9.0]))

    def test_broadcast_row_over_matrix(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([10.0, 20.0, 30.0])
        expected = np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])
        assert_allclose(broadcast_add(a, b), expected)

    def test_broadcast_column_over_row(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([[0.0], [10.0]])
        expected = np.array([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
        assert_allclose(broadcast_add(a, b), expected)

    def test_scalar_broadcast(self):
        a = np.array([1.0, 2.0, 3.0])
        assert_allclose(broadcast_add(a, np.array(5.0)), np.array([6.0, 7.0, 8.0]))

    def test_invalid_list_input(self):
        with pytest.raises(ValueError, match="np.ndarray"):
            broadcast_add([1, 2, 3], np.array([1, 2, 3]))

    def test_invalid_incompatible_shapes(self):
        with pytest.raises(ValueError, match="broadcastable"):
            broadcast_add(np.zeros((3, 4)), np.zeros((2, 5)))

    def test_output_dtype_preserved(self):
        result = broadcast_add(np.array([1], dtype=np.int32),
                               np.array([1.0], dtype=np.float32))
        assert result.dtype in (np.float32, np.float64)

class TestRowColMeanDiff:
    def test_known_result_2x3(self):
        m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected = np.array([[-1.25, -0.75, -0.25], [0.25, 0.75, 1.25]])
        assert_allclose(row_col_mean_diff(m), expected, atol=1e-12)

    def test_uniform_matrix_gives_zeros(self):
        assert_allclose(row_col_mean_diff(np.full((4, 5), 7.0)),
                        np.zeros((4, 5)), atol=1e-12)

    def test_single_element_matrix(self):
        assert_allclose(row_col_mean_diff(np.array([[42.0]])),
                        np.array([[0.0]]), atol=1e-12)

    def test_output_shape_preserved(self):
        m = np.random.default_rng(0).standard_normal((7, 11))
        assert row_col_mean_diff(m).shape == (7, 11)

    def test_invalid_1d_array(self):
        with pytest.raises(ValueError):
            row_col_mean_diff(np.array([1.0, 2.0, 3.0]))

    def test_invalid_plain_list(self):
        with pytest.raises(ValueError):
            row_col_mean_diff([[1, 2], [3, 4]])

class TestNormalizeColumns:
    def test_zero_mean_per_column(self):
        m = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        assert_allclose(normalize_columns(m).mean(axis=0), [0.0, 0.0], atol=1e-12)

    def test_unit_std_per_column(self):
        m = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        assert_allclose(normalize_columns(m).std(axis=0), [1.0, 1.0], atol=1e-12)

    def test_constant_column_becomes_zeros(self):
        m = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        out = normalize_columns(m)
        assert_allclose(out[:, 0], [0.0, 0.0, 0.0], atol=1e-12)
        assert not np.any(np.isnan(out))

    def test_known_values_single_column(self):
        m = np.array([[-1.0], [0.0], [1.0]])
        std = np.sqrt(2.0 / 3.0)
        assert_allclose(normalize_columns(m)[:, 0],
                        [-1.0 / std, 0.0, 1.0 / std], atol=1e-12)

    def test_invalid_1d_array(self):
        with pytest.raises(ValueError):
            normalize_columns(np.array([1.0, 2.0, 3.0]))

    def test_invalid_single_row(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            normalize_columns(np.array([[1.0, 2.0, 3.0]]))

    def test_output_dtype_is_float64(self):
        m = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        assert normalize_columns(m).dtype == np.float64

class TestSlidingWindowSum:
    def test_basic_window_3(self):
        assert_allclose(sliding_window_sum(np.array([1.,2.,3.,4.,5.]), 3),
                        [6., 9., 12.], atol=1e-12)

    def test_window_equals_length(self):
        assert_allclose(sliding_window_sum(np.array([1.,2.,3.]), 3), [6.], atol=1e-12)

    def test_window_1_identity(self):
        arr = np.array([4., 5., 6.])
        assert_allclose(sliding_window_sum(arr, 1), arr, atol=1e-12)

    def test_output_length_formula(self):
        arr = np.arange(10, dtype=float)
        for w in [1, 2, 5, 10]:
            assert len(sliding_window_sum(arr, w)) == len(arr) - w + 1

    def test_negative_values(self):
        assert_allclose(sliding_window_sum(np.array([-3.,1.,-1.,2.]), 2),
                        [-2., 0., 1.], atol=1e-12)

    def test_invalid_window_too_large(self):
        with pytest.raises(ValueError):
            sliding_window_sum(np.array([1., 2.]), 5)

    def test_invalid_window_zero(self):
        with pytest.raises(ValueError):
            sliding_window_sum(np.array([1., 2., 3.]), 0)

    def test_invalid_2d_input(self):
        with pytest.raises(ValueError):
            sliding_window_sum(np.array([[1., 2.]]), 1)

class TestPairwiseCosineSimilarity:
    def test_orthogonal_vectors(self):
        X = np.array([[1., 0.], [0., 1.]])
        assert_allclose(pairwise_cosine_similarity(X),
                        np.array([[1.,0.],[0.,1.]]), atol=1e-12)

    def test_identical_vectors(self):
        X = np.array([[3., 4.], [3., 4.]])
        assert_allclose(pairwise_cosine_similarity(X), np.ones((2,2)), atol=1e-12)

    def test_opposite_vectors(self):
        X = np.array([[1., 0.], [-1., 0.]])
        assert_allclose(pairwise_cosine_similarity(X),
                        np.array([[1.,-1.],[-1.,1.]]), atol=1e-12)

    def test_matrix_is_symmetric(self):
        X = np.random.default_rng(42).standard_normal((6, 8))
        result = pairwise_cosine_similarity(X)
        assert_allclose(result, result.T, atol=1e-12)

    def test_diagonal_is_one(self):
        X = np.random.default_rng(0).standard_normal((5, 3))
        assert_allclose(np.diag(pairwise_cosine_similarity(X)), np.ones(5), atol=1e-12)

    def test_zero_vector_gives_zero_row(self):
        X = np.array([[0.,0.],[1.,0.],[0.,1.]])
        assert_allclose(pairwise_cosine_similarity(X)[0,:], [0.,0.,0.], atol=1e-12)

    def test_values_clipped_to_valid_range(self):
        X = np.random.default_rng(7).standard_normal((8, 4))
        result = pairwise_cosine_similarity(X)
        assert np.all(result >= -1.0 - 1e-12) and np.all(result <= 1.0 + 1e-12)

    def test_output_shape(self):
        X = np.random.default_rng(1).standard_normal((5, 10))
        assert pairwise_cosine_similarity(X).shape == (5, 5)

    def test_invalid_1d_input(self):
        with pytest.raises(ValueError):
            pairwise_cosine_similarity(np.array([1., 2., 3.]))
