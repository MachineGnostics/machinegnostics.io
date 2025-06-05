# mean_absolute_error: Mean Absolute Error (MAE) Metric

The `mean_absolute_error` function computes the mean absolute error (MAE) between true and predicted values. MAE is a fundamental regression metric that measures the average magnitude of errors in a set of predictions, without considering their direction.

---

## Overview

Mean Absolute Error is defined as the average of the absolute differences between actual and predicted values.

MAE is widely used in regression analysis to quantify how close predictions are to the actual outcomes. Lower MAE values indicate better model performance.

---

## Parameters

| Parameter | Type         | Description                        |
|-----------|--------------|------------------------------------|
| `y_true`  | array-like   | True values (targets).             |
| `y_pred`  | array-like   | Predicted values.                  |

---

## Returns

- **float**  
  The average absolute difference between actual and predicted values.

---

## Raises

- **TypeError**  
  If `y_true` or `y_pred` are not array-like (list, tuple, or numpy array).
- **ValueError**  
  If inputs have mismatched shapes or are empty.

---

## Example Usage

```python
from machinegnostics.metrics import mean_absolute_error

# Example 1: Using lists
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_absolute_error(y_true, y_pred))  # Output: 0.5

# Example 2: Using numpy arrays
import numpy as np
y_true = np.array([1, 2, 3])
y_pred = np.array([1, 2, 2])
print(mean_absolute_error(y_true, y_pred))  # Output: 0.3333333333333333
```

---

## Notes

- The function supports input as lists, tuples, or numpy arrays.
- Both `y_true` and `y_pred` must have the same shape and must not be empty.
- MAE is robust to outliers but does not penalize large errors as strongly as mean squared error (MSE).

---
