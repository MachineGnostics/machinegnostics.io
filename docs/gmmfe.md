# gmmfe: Geometric Mean of Model Fit Error (GMMFE) Metric

The `gmmfe` function computes the Geometric Mean of Model Fit Error (GMMFE), a robust metric for evaluating the average relative error between observed data and model predictions on a logarithmic scale. GMMFE is especially useful for datasets with a wide range of values or when the data is multiplicative in nature.

---

## Overview

GMMFE quantifies the average multiplicative error between the true and predicted values, making it less sensitive to outliers and scale differences than classical metrics. It is one of the three core criteria (alongside RobR2 and DivI) for evaluating model performance in the Machine Gnostics framework.

Mathematically, GMMFE is defined as:

$$
\text{GMMFE} = \exp\left( \frac{1}{N} \sum_{i=1}^N \left| \log\left(\frac{y_i}{\hat{y}_i}\right) \right| \right)
$$

where:

- $y_i$ is the observed value,
- $\hat{y}_i$ is the fitted (predicted) value,
- $N$ is the number of data points.

A lower GMMFE indicates a better fit, as it means the geometric mean of the relative errors is smaller.

---

## Interpretation

- **Lower GMMFE**: Indicates smaller average multiplicative errors and a better model fit.
- **Higher GMMFE**: Indicates larger average multiplicative errors and a poorer fit.

GMMFE is particularly valuable when comparing models across datasets with different scales or when the error distribution is multiplicative.

---

## Parameters

| Parameter | Type       | Description                                                     |
| --------- | ---------- | --------------------------------------------------------------- |
| `y`     | np.ndarray | Observed data (ground truth). 1D array of numerical values.     |
| `y_fit` | np.ndarray | Fitted data (model predictions). 1D array, same shape as `y`. |

---

## Returns

- **float**
  The computed Geometric Mean of Model Fit Error (GMMFE) value.

---

## Raises

- **ValueError**
  - If `y` and `y_fit` do not have the same shape.
  - If `y` or `y_fit` are not 1D arrays.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import gmmfe

y = np.array([1.0, 2.0, 3.0, 4.0])
y_fit = np.array([1.1, 1.9, 3.2, 3.8])
result = gmmfe(y, y_fit)
print(result)  # Output: 0.06666666666666667
```

---

## Notes

- GMMFE is calculated using the weighted geometric mean of the relative errors.
- It is robust to outliers and scale differences, making it suitable for a wide range of regression problems.
- In the Machine Gnostics framework, GMMFE is used alongside RobR2 and DivI to provide a comprehensive evaluation of model performance.
- The overall evaluation metric can be computed as:

  $$
  \text{EvalMet} = \frac{\text{RobR2}}{\text{GMMFE} \cdot \text{DivI}}
  $$

  where a higher EvalMet indicates better model performance.

---

## References

- Kovanic P., Humber M.B (2015) *The Economics of Information - Mathematical Gnostics for Data Analysis*

---

## License

Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

---
