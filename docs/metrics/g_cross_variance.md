# cross_covariance: Gnostic Cross-Covariance Metric

The `cross_covariance` function computes the Gnostic cross-covariance between two data samples. This metric uses gnostic theory to provide robust, assumption-free estimates of relationships between datasets, even in the presence of noise or outliers.

---

## Overview

Gnostic cross-covariance generalizes classical covariance by leveraging irrelevance measures from gnostic theory:

- **Estimating irrelevances** are aggregated as trigonometric sines (case `'i'`).
- **Quantifying irrelevances** are aggregated as hyperbolic sines (case `'j'`).

Both approaches converge to linear error in cases of weak uncertainty, but provide robust diagnostics in challenging data scenarios. The metric is computed as the mean product of irrelevances between two data samples.

---

## Parameters

| Parameter   | Type       | Description                                                                      |
| ----------- | ---------- | -------------------------------------------------------------------------------- |
| `data_1`  | np.ndarray | First data sample (1D numpy array, no NaN/Inf).                                  |
| `data_2`  | np.ndarray | Second data sample (1D numpy array, no NaN/Inf).                                 |
| `case`    | str        | Geometry type:`'i'` for estimation, `'j'` for quantifying. Default: `'i'`. |
| `verbose` | bool       | If True, enables detailed logging for debugging. Default:`False`.              |

---

## Returns

- **float**
  The Gnostic cross-covariance between the two data samples.

---

## Raises

- **ValueError**
  If input arrays are not the same length, are empty, contain NaN/Inf, are not 1D, or if `case` is not `'i'` or `'j'`.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import cross_covariance

# Example 1: Compute cross-covariance for two simple datasets
data_1 = np.array([1, 2, 3, 4, 5])
data_2 = np.array([5, 4, 3, 2, 1])
covar = cross_covariance(data_1, data_2, case='i', verbose=False)
print(f"Cross-Covariance (case='i'): {covar}")

# Example 2: Using quantifying geometry
covar_j = cross_covariance(data_1, data_2, case='j', verbose=True)
print(f"Cross-Covariance (case='j'): {covar_j}")

# Example 3: Handle invalid input
data_invalid = np.array([1, np.nan, 3, 4, 5])
try:
    covar = cross_covariance(data_invalid, data_2, case='i')
except ValueError as e:
    print(f"Error: {e}")
```

---

## Notes

- The metric is robust to data uncertainty, noise, and outliers.
- Input data must be preprocessed and cleaned for optimal results.
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
