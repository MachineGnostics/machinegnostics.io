# evalMet: Composite Evaluation Metric

The `evalMet` function computes the Evaluation Metric (EvalMet), a composite score that combines three robust criteria—Robust R-squared (RobR2), Geometric Mean of Model Fit Error (GMMFE), and Divergence Information (DivI)—to provide a comprehensive assessment of model performance.

---

## Overview

EvalMet is designed to quantify the overall quality of a model fit by integrating three complementary metrics:

- **RobR2**: Measures the proportion of variance explained by the model, robust to outliers.
- **GMMFE**: Captures the average multiplicative fitting error on a logarithmic scale.
- **DivI**: Quantifies the divergence in information content between the observed data and the model fit.

The combined metric is calculated as:

$$
\text{EvalMet} = \frac{\text{RobR2}}{\text{GMMFE} \cdot \text{DivI}}
$$

A higher EvalMet value indicates a better model fit, balancing explained variance, error magnitude, and information divergence.

---

## Parameters

| Parameter | Type       | Default | Description                                                                                               |
| --------- | ---------- | ------- | --------------------------------------------------------------------------------------------------------- |
| `y`     | np.ndarray | —      | Observed data (ground truth). 1D array of numerical values.                                               |
| `y_fit` | np.ndarray | —      | Fitted data (model predictions). 1D array, same shape as `y`.                                           |
| `w`     | np.ndarray | None    | Optional weights for data points. 1D array, same shape as `y`. If not provided, equal weights are used. |

---

## Returns

- **float**
  The computed Evaluation Metric (EvalMet) value.

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
from machinegnostics.metrics import evalMet

y = np.array([1.0, 2.0, 3.0, 4.0])
y_fit = np.array([1.1, 1.9, 3.2, 3.8])
result = evalMet(y, y_fit)
print(result)
```

---

## Notes

- **EvalMet** is most informative when used to compare multiple models or methods on the same dataset.
- The metric is robust to outliers and non-Gaussian data due to its use of gnostic algebra.
- EvalMet is especially useful in benchmarking and model selection scenarios, as it integrates multiple aspects of fit quality into a single score.

---

## References

- Kovanic P., Humber M.B (2015) *The Economics of Information - Mathematical Gnostics for Data Analysis*

---

## License

Machine Gnostics - Machine Gnostics Library   
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

---
