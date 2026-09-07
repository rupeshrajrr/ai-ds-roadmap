import timeit
import numpy as np
from numpy_exercises import sliding_window_sum

def naive_sliding_window_sum(arr, window):
    """Standard Python for-loop approach."""
    result = []
    # Using standard Python lists and sum()
    for i in range(len(arr) - window + 1):
        result.append(sum(arr[i:i+window]))
    return result

def run_benchmark():
    print("Preparing data...")
    # 100,000 random floats
    np_arr = np.random.rand(100_000)
    py_list = np_arr.tolist()
    window_size = 1000

    print(f"Benchmarking Sliding Window Sum (Size: {len(np_arr):,}, Window: {window_size})")
    print("-" * 60)

    # Benchmark standard Python (Running 10 times)
    py_time = timeit.timeit(lambda: naive_sliding_window_sum(py_list, window_size), number=10)
    print(f"Standard Python Loop: {py_time:.4f} seconds")

    # Benchmark NumPy (Running 10 times)
    np_time = timeit.timeit(lambda: sliding_window_sum(np_arr, window_size), number=10)
    print(f"NumPy Vectorized:     {np_time:.4f} seconds")
    
    print("-" * 60)
    print(f"NumPy is {py_time / np_time:,.0f}x faster!")

if __name__ == "__main__":
    run_benchmark()
