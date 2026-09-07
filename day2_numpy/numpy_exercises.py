from __future__ import annotations
import numpy as np

# --- Set 1 Functions ---

def broadcast_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("Both inputs must be np.ndarray instances.")
    try:
        np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        raise ValueError("Shapes are not broadcastable.")
    return a + b

def row_col_mean_diff(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2-D np.ndarray.")
    mat = matrix.astype(np.float64)
    row_means = mat.mean(axis=1, keepdims=True)
    col_means = mat.mean(axis=0, keepdims=True)
    return mat - (row_means + col_means) / 2.0

def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2-D np.ndarray.")
    if matrix.shape[0] < 2:
        raise ValueError("Matrix must have at least 2 rows to standardise.")
    mat = matrix.astype(np.float64)
    means = mat.mean(axis=0)
    stds = mat.std(axis=0)
    safe_stds = np.where(stds == 0.0, 1.0, stds)
    normalised = (mat - means) / safe_stds
    normalised[:, stds == 0.0] = 0.0
    return normalised

def sliding_window_sum(arr: np.ndarray, window: int) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError("arr must be a 1-D np.ndarray.")
    if not isinstance(window, int) or window < 1 or window > len(arr):
        raise ValueError("window must be an int.")
    prefix = np.empty(len(arr) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(arr, out=prefix[1:])
    return prefix[window:] - prefix[:-window]

def pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2-D np.ndarray.")
    X_f = X.astype(np.float64)
    norms = np.linalg.norm(X_f, axis=1, keepdims=True)
    zero_rows = (norms[:, 0] == 0.0)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    X_norm = X_f / safe_norms
    X_norm[zero_rows] = 0.0
    sim = X_norm @ X_norm.T
    return np.clip(sim, -1.0, 1.0)

# --- Set 2 Functions ---

def min_max_scale(arr: np.ndarray, feature_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return np.full_like(arr, feature_range[0], dtype=np.float64)
    scaled = (arr - min_val) / (max_val - min_val)
    return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

def standardise(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0.0:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mean) / std

def moving_average_1d(arr: np.ndarray, window: int) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError("Input must be a 1-D numpy ndarray.")
    if not isinstance(window, int) or window < 1 or window > len(arr):
        raise ValueError("Invalid window size.")
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

def one_hot_encode(arr: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if num_classes is None:
        num_classes = int(np.max(arr)) + 1
    res = np.zeros((arr.size, num_classes), dtype=np.float64)
    res[np.arange(arr.size), arr.ravel()] = 1.0
    return res.reshape(arr.shape + (num_classes,))

def softmax(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    shift_x = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=-1, keepdims=True)
