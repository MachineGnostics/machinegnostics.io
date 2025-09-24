# correlation: Gnostic Correlation Metric

The `correlation` function computes the Gnostic correlation coefficient between two data samples. This metric is based on gnostic theory, providing robust and interpretable estimates of correlation that are resilient to uncertainty, noise, and outliers.

---

## Overview

Gnostic correlation generalizes classical correlation by leveraging irrelevance measures from gnostic theory.

- **Estimating irrelevances** are aggregated as trigonometric sines (case `'i'`).
- **Quantifying irrelevances** are aggregated as hyperbolic sines (case `'j'`).

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
