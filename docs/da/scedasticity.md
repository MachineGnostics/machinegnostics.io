# DataScedasticity: Gnostic Homoscedasticity and Heteroscedasticity Test (Machine Gnostics)

The `DataScedasticity` class provides a gnostic approach to testing for homoscedasticity and heteroscedasticity in data, using gnostic variance and gnostic linear regression. Unlike classical statistical tests, this method is based on the principles of the Machine Gnostics framework and is designed for robust, interpretable diagnostics.

---

## Overview

DataScedasticity analyzes the spread of residuals from a gnostic linear regression model by splitting the data at the median of the independent variable and comparing the gnostic variances of squared residuals in each half. This approach is fundamentally different from classical tests (e.g., Breusch-Pagan, White's test), focusing on gnostic variance and regression rather than least squares and probabilistic assumptions.

- **Gnostic Variance:** Measures uncertainty and spread according to gnostic principles.
- **Gnostic Regression:** Uses a gnostic linear regression model, not standard least squares.
- **Diagnostic Philosophy:** Not a formal statistical test, but a robust diagnostic for gnostic data analysis.
- **Split Residuals:** Compares variance in residuals before and after the median of the independent variable.

---

## Key Features

- **Gnostic variance and regression for scedasticity analysis**
- **No reliance on classical statistical assumptions**
- **Robust to outliers and non-Gaussian data**
- **Variance ratio calculation for split residuals**
- **Clear homoscedastic/heteroscedastic decision**
- **Detailed logging and parameter tracking**

---

## Parameters

| Parameter                | Type                  | Default   | Description                                                      |
|--------------------------|-----------------------|-----------|------------------------------------------------------------------|
| `scale`                  | str/int/float         | 'auto'    | Scale parameter for regression                                   |
| `max_iter`               | int                   | 100       | Maximum iterations for regression optimization                   |
| `tol`                    | float                 | 0.001     | Tolerance for regression convergence                             |
| `mg_loss`                | str                   | 'hi'      | Loss function for gnostic regression                             |
| `early_stopping`         | bool                  | True      | Enable early stopping in regression                              |
| `verbose`                | bool                  | False     | Print detailed logs and diagnostics                              |
| `data_form`              | str                   | 'a'       | Data form: 'a' (additive), 'm' (multiplicative)                  |
| `gnostic_characteristics`| bool                  | True      | Use gnostic characteristics in regression                        |
| `history`                | bool                  | True      | Track regression history                                         |

---

## Attributes

- **x**: `np.ndarray`  
  Independent variable data.
- **y**: `np.ndarray`  
  Dependent variable data.
- **model**: `LinearRegressor`  
  Gnostic linear regression model.
- **residuals**: `np.ndarray`  
  Residuals from the fitted model.
- **params**: `dict`  
  Stores calculated variances and variance ratio.
- **variance_ratio**: `float`  
  Ratio of gnostic variances between data splits.
- **is_homoscedastic**: `bool`  
  True if data is homoscedastic under gnostic test, else False.

---

## Methods

### `fit(x, y)`

Fits the gnostic linear regression model to the data and assesses scedasticity.

- **x**: `np.ndarray`  
  Independent variable data.
- **y**: `np.ndarray`  
  Dependent variable data.

**Returns:**  
`bool` — True if data is homoscedastic under the gnostic test, False if heteroscedastic.

---

## Example Usage

```python
import numpy as np
from machinegnostics.magcal import DataScedasticity

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.1, 4.2, 6.1, 8.3, 10.2, 12.1, 14.2, 16.1, 18.2, 20.1])

sced = DataScedasticity()
is_homo = sced.fit(x, y)
print(f"Is data homoscedastic? {is_homo}")
print(f"Variance ratio: {sced.variance_ratio}")
```

---

## Notes

- This is not a standard statistical test; results may differ from classical methods.
- Gnostic variance and regression are designed for robust, interpretable diagnostics.
- For more details on gnostic variance and regression, refer to the Machine Gnostics documentation.

---

**Author:** Nirmal Parmar  
**Date:** 2025-09-24

---