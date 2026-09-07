# Day 2 — Python & NumPy Mastery

Five fully-vectorised NumPy functions covering broadcasting, double-centring,
column normalisation, sliding window aggregation, and pairwise cosine similarity.
Zero Python loops. 32 pytest tests (all passing).

## Quick start
\\\ash
# from ai-ds-roadmap/ root with venv active:
cd day2_numpy
python -m pytest tests/ -q       # run all tests
python -m pyflakes numpy_exercises.py  # lint check
\\\

## Functions
| Function | Concept |
|---|---|
| \roadcast_add\ | NumPy broadcasting rules |
| \ow_col_mean_diff\ | keepdims + double-centring |
| \
ormalize_columns\ | Axis-wise reduction + safe division |
| \sliding_window_sum\ | Prefix-sum (cumsum) trick |
| \pairwise_cosine_similarity\ | BLAS matrix multiply (Gram matrix) |
