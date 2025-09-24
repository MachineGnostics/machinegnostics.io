# variance: Gnostic Variance Metric

The `variance` function computes the Gnostic variance of a data sample. This metric uses gnostic theory to provide robust, assumption-free estimates of data variability, leveraging irrelevance measures for deeper insight into uncertainty and structure.

---

## Overview

Gnostic variance generalizes classical variance by using irrelevance measures:

- **Case `'i'`**: Estimates variance using ELDF (Empirical Likelihood Distribution Function).
- **Case `'j'`**: Quantifies variance using QLDF (Quantile Likelihood Distribution Function).

Both approaches are robust to outliers and non-normal data, providing reliable diagnostics in challenging scenarios.

---

## Parameters

| Parameter       | Type       | Description                                                                      | Default   |
| --------------- | ---------- | -------------------------------------------------------------------------------- | --------- |
| `data`        | np.ndarray | Input data array (1D, no NaN/Inf).                                               | Required  |
| `case`        | str        | `'i'` for estimating variance (ELDF), `'j'` for quantifying variance (QLDF). | `'i'`   |
| `S`           | float      | Scaling parameter for ELDF/QLDF.                                                 | `1`     |
| `z0_optimize` | bool       | Whether to optimize z0 in ELDF/QLDF.                                             | `True`  |
| `data_form`   | str        | Data form for ELDF/QLDF:`'a'` for additive, `'m'` for multiplicative.        | `'a'`   |
| `tolerance`   | float      | Tolerance for ELDF fitting.                                                      | `1e-6`  |
| `verbose`     | bool       | If True, enables detailed logging for debugging.                                 | `False` |

---

## Returns

- **float**
  The Gnostic variance of the data.

---

## Raises

- **TypeError**If input is not a numpy array.
- **ValueError**
  If input is not 1D, is empty, contains NaN/Inf, or if `case` is not `'i'` or `'j'`.

---

## Example Usage

```python
import machinegnostics as mg
import numpy as np

# Example 1: Compute gnostic variance (default case)
data = np.array([1, 2, 3, 4, 5])
var = mg.variance(data)
print(var)

# Example 2: Quantifying variance with QLDF
var_j = mg.variance(data, case='j')
print(var_j)
```

---

## Notes

- The function uses ELDF or QLDF to compute irrelevance values, which are then squared and averaged.
- Input data must be 1D, cleaned, and free of NaN/Inf.
- The metric is robust to outliers and non-normal data, providing more reliable diagnostics than classical variance.
- Scaling (`S`), optimization (`z0_optimize`), and data form (`data_form`) parameters allow for flexible analysis.

---

## Gnostic vs. Classical Variance

> **Note:**
> Unlike classical variance metrics that use statistical means, the Gnostic variance is computed using irrelevance measures from gnostic theory. This approach is assumption-free and designed to reveal the true diagnostic properties of your data.

---

**Authors:** Nirmal Parmar  
**Date:** 2025-09-24

---
