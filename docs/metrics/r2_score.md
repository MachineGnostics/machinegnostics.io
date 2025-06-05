# robr2: Robust R-squared (RobR2) Metric

The `robr2` function computes the Robust R-squared (RobR2) value for evaluating the goodness of fit between observed data and model predictions. Unlike the classical R-squared metric, RobR2 is robust to outliers and incorporates sample weights, making it ideal for noisy or irregular datasets.

---

## Overview

Robust R-squared (RobR2) measures the proportion of variance in the observed data explained by the fitted data, while reducing sensitivity to outliers. This is achieved by using a weighted formulation, which allows for more reliable model evaluation in real-world scenarios where data may not be perfectly clean.

---

## Formula

$$
\text{RobR2} = 1 - \frac{\sum_i w_i (e_i - \bar{e})^2}{\sum_i w_i (y_i - \bar{y})^2}
$$

Where:

- $e_i = y_i - \hat{y}_i$ (residuals)
- $\bar{e}$ = weighted mean of residuals
- $\bar{y}$ = weighted mean of observed data
- $w_i$ = weight for each data point

If weights are not provided, equal weights are assumed.

---

## Parameters

| Parameter | Type               | Description                                                                                       |
| --------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| `y`     | np.ndarray         | Observed data (ground truth). 1D array of numerical values.                                       |
| `y_fit` | np.ndarray         | Fitted data (model predictions). 1D array, same shape as `y`.                                   |
| `w`     | np.ndarray or None | Optional weights for data points. 1D array, same shape as `y`. If None, equal weights are used. |

---

## Returns

- **float**
  The computed Robust R-squared (RobR2) value. Ranges from 0 (no explanatory power) to 1 (perfect fit).

---

## Raises

- **ValueError**
  - If `y` and `y_fit` do not have the same shape.
  - If `w` is provided and does not have the same shape as `y`.
  - If `y` or `y_fit` are not 1D arrays.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import robr2

y = np.array([1.0, 2.0, 3.0, 4.0])
y_fit = np.array([1.1, 1.9, 3.2, 3.8])
w = np.array([1.0, 1.0, 1.0, 1.0])

result = robr2(y, y_fit, w)
print(result)  # Example output: 0.98
```

---

## Comparison with Classical R-squared

- **Classical R-squared**: Assumes equal weights and is sensitive to outliers.
- **RobR2**: Incorporates weights and is robust to outliers, making it more reliable for datasets with irregularities or noise.

---

## References

- Kovanic P., Humber M.B (2015) *The Economics of Information - Mathematical Gnostics for Data Analysis*, Chapter 19

---

## Notes

- If weights are not provided, the metric defaults to equal weighting for all data points.
- RobR2 is particularly useful for robust regression and model evaluation in the presence of outliers.

---
