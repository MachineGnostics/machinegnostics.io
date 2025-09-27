
# Gnostic Cross-Covariance Metric

The `cross_covariance` function computes the Gnostic cross-covariance between two data samples. This metric uses gnostic theory to provide robust, assumption-free estimates of relationships between datasets, even in the presence of noise or outliers. Argument names have changed: use `X` and `y` for input arrays.

---

## Overview

Gnostic cross-covariance generalizes classical covariance by leveraging irrelevance measures from gnostic theory:

- **Estimating irrelevances** are aggregated as trigonometric sines (case `'i'`).
- **Quantifying irrelevances** are aggregated as hyperbolic sines (case `'j'`).

Both approaches converge to linear error in cases of weak uncertainty, but provide robust diagnostics in challenging data scenarios. The metric is computed as the mean product of irrelevances between two data samples.

---


## Parameters

| Parameter | Type       | Description                                                                                 |
|-----------|------------|---------------------------------------------------------------------------------------------|
| `X`       | np.ndarray | Feature data sample (1D numpy array, no NaN/Inf). For multi-column X, pass each column separately. |
| `y`       | np.ndarray | Target data sample (1D numpy array, no NaN/Inf).                                            |
| `case`    | str        | Geometry type: `'i'` for estimation (EGDF), `'j'` for quantifying (QGDF). Default: `'i'`.   |
| `verbose` | bool       | If True, enables detailed logging for debugging. Default: `False`.                          |

---

## Returns

- **float**
  The Gnostic cross-covariance between the two data samples.

---

## Raises

- **ValueError**
  If input arrays are not the same length, are empty, contain NaN/Inf, are not 1D, or if `case` is not `'i'` or `'j'`.

---

covar = cross_covariance(data_1, data_2, case='i', verbose=False)

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import cross_covariance

# Example 1: Compute cross-covariance for two simple datasets
X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
covar = cross_covariance(X, y, case='i', verbose=False)
print(f"Cross-Covariance (case='i'): {covar}")

# Example 2: For multi-column X
X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
y = np.array([5, 4, 3, 2, 1])
for i in range(X.shape[1]):
  covar = cross_covariance(X[:, i], y)
  print(f"Cross-Covariance for column {i}: {covar}")

# Example 3: Handle invalid input
X_invalid = np.array([1, np.nan, 3, 4, 5])
try:
  covar = cross_covariance(X_invalid, y, case='i')
except ValueError as e:
  print(f"Error: {e}")
```

---


## Notes

- `X` must be a 1D numpy array (single column). For multi-column X, pass each column separately (e.g., `X[:, i]`).
- `y` must be a 1D numpy array.
- Both arrays must be of the same length, with no NaN or Inf values.
- The metric is robust to data uncertainty and provides meaningful estimates even in the presence of noise or outliers.
- Ensure that the input data is preprocessed and cleaned for optimal results.
- If data homogeneity is not met, the function automatically adjusts scale parameters for better reliability.
- The Gnostic cross-covariance uses irrelevance measures rather than classical means, providing deeper insight into relationships between datasets.
- Supports both estimation and quantification geometries for flexible analysis.

---

## Gnostic vs. Classical Covariance

> **Note:**
> Unlike classical covariance metrics that rely on statistical means, the Gnostic cross-covariance uses irrelevance measures derived from gnostic theory. This approach is assumption-free and designed to reveal true relationships, even in the presence of outliers or non-normal distributions.

---

**Author:** Nirmal Parmar	  
**Date:** 2025-09-24

---
