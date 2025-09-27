# correlation: Gnostic Correlation Metric

The Gnostic correlation metric provides a robust, assumption-free measure of association between two data samples. Unlike traditional correlation coefficients, which rely on statistical means and are sensitive to outliers, the Gnostic approach leverages the principles of Mathematical Gnostics to deliver reliable results even in the presence of noise, outliers, or non-Gaussian data.

---

## Overview

The `correlation` function computes the Gnostic correlation coefficient between two arrays, using either estimation geometry (EGDF) or quantifying geometry (QGDF). This metric is designed to be resilient to data uncertainty and is meaningful for both small and large datasets.

---

## Usage

```python
# Example 1: Compute correlation for two simple datasets
import numpy as np
from machinegnostics.metrics import correlation

X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
corr = correlation(X, y, case='i', verbose=False)
print(f"Correlation (case='i'): {corr}")

# Example 2: For multi-column X
X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
y = np.array([5, 4, 3, 2, 1])
for i in range(X.shape[1]):
    corr = correlation(X[:, i], y)
    print(f"Correlation for column {i}: {corr}")
```

---

## Parameters

- **X** (`np.ndarray`): Feature data sample. Should be a 1D array or a single column from a 2D array, with no NaN or Inf values.
- **y** (`np.ndarray`): Target data sample. Must be a 1D array, with no NaN or Inf values.
- **case** (`str`, default `'i'`):
  - `'i'`: Use estimation geometry (EGDF)
  - `'j'`: Use quantifying geometry (QGDF)
- **verbose** (`bool`, default `False`): If `True`, enables detailed logging for debugging.

---

## Returns

- **float**: The Gnostic correlation coefficient between the two data samples.

---

## Notes & Best Practices

- If `X` has more than one column, pass each column separately (e.g., `X[:, i]`).
- Both `X` and `y` must be 1D arrays of the same length, with no missing or infinite values.
- The metric is robust to outliers and provides meaningful estimates even for small or noisy datasets.
- If data homogeneity is not met, the function will adjust parameters and issue a warning for best results.
- For optimal results, ensure your data is preprocessed and cleaned.

---

## Error Handling

The function will raise a `ValueError` if:

- The input arrays are not of the same length, are empty, or contain NaN/Inf values.
- The `case` argument is not `'i'` or `'j'`.

---

## References

- See the [Machine Gnostics documentation](https://machinegnostics.info/) for more details on Gnostic metrics and their theoretical foundation.

---

**Author:** Dr. Nirmal Parmar
**Project:** Machine Gnostics

Both approaches converge to linear error in cases of weak uncertainty, but provide robust diagnostics in challenging data scenarios. The metric is designed to reflect true relationships in your data, even when classical methods may fail.

---

## Parameters

| Parameter   | Type       | Description                                                                                    |
| ----------- | ---------- | ---------------------------------------------------------------------------------------------- |
| `data_1`  | np.ndarray | First data sample (1D numpy array, no NaN/Inf).                                                |
| `data_2`  | np.ndarray | Second data sample (1D numpy array, no NaN/Inf).                                               |
| `case`    | str        | Geometry type:`'i'` for estimation (EGDF), `'j'` for quantifying (QGDF). Default: `'i'`. |
| `verbose` | bool       | If True, enables detailed logging for debugging. Default:`False`.                            |

---

## Returns

- **float**
  The Gnostic correlation coefficient between the two data samples.

---

## Raises

- **ValueError**
  If input arrays are not the same length, are empty, contain NaN/Inf, are not 1D, or if `case` is not `'i'` or `'j'`.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import correlation

# Example 1: Compute correlation for two simple datasets
data_1 = np.array([1, 2, 3, 4, 5])
data_2 = np.array([5, 4, 3, 2, 1])
corr = correlation(data_1, data_2, case='i', verbose=False)
print(f"Correlation (case='i'): {corr}")

# Example 2: Using quantifying geometry
corr_j = correlation(data_1, data_2, case='j', verbose=True)
print(f"Correlation (case='j'): {corr_j}")
```

---

## Notes

- The metric is robust to data uncertainty, noise, and outliers.
- Input data must be preprocessed and cleaned for optimal results.
- If data homogeneity is not met, the function automatically adjusts scale parameters for better reliability.
- The Gnostic correlation uses irrelevance measures rather than classical means, providing deeper insight into data relationships.
- Supports both estimation and quantification geometries for flexible analysis.

---

## Gnostic vs. Classical Correlation

> **Note:**
> Unlike classical correlation metrics that rely on statistical means and linear relationships, the Gnostic correlation uses irrelevance measures derived from gnostic theory. This approach is assumption-free and designed to reveal true data relationships, even in the presence of outliers or non-normal distributions.

---

**Author:** Nirmal Parmar	
**Date:** 2025-09-24

---
