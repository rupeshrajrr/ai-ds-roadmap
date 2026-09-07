import numpy as np

def min_max_scale(X: np.ndarray) -> np.ndarray:
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    range_vals = max_vals - min_vals
    safe_range = np.where(range_vals == 0, 1.0, range_vals)
    scaled = (X - min_vals) / safe_range
    return np.where(range_vals == 0, 0.0, scaled)

def standardise(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    res = (X - mean) / safe_std
    return np.where(std == 0, 0.0, res)

def moving_average_1d(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 0 or w > len(x): raise ValueError("Invalid window.")
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[w:] - cumsum[:-w]) / w

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[labels]

def softmax(X: np.ndarray) -> np.ndarray:
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)
