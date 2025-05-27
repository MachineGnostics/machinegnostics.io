# gcorrelation: Gnostic Correlation Metric

The `gcorrelation` function computes the Gnostic correlation coefficient between two data samples using robust irrelevance-based weighting. This metric provides a robust alternative to the classical Pearson correlation, making it less sensitive to outliers and non-normal data distributions.

---

## Overview

Gnostic correlation leverages irrelevance functions to construct robust weights for each data point, following the gnostic framework described by Kovanic & Humber (2015). This approach allows for a more reliable measure of association between variables, especially in the presence of noise or outliers.

- **Robust to outliers:** Uses irrelevance-based weighting.
- **No normality assumption:** Works well with non-Gaussian data.
- **Flexible:** Supports both 1D and 2D data (column-wise correlation).

---

## Parameters

| Parameter | Type         | Description                                                                 |
|-----------|-------------|-----------------------------------------------------------------------------|
| `data_1`  | np.ndarray, pandas Series, or DataFrame | First data sample (1D or 2D). Each column is treated as a variable. |
| `data_2`  | np.ndarray, pandas Series, or DataFrame | Second data sample (must have same number of rows as `data_1`).     |

---

## Returns

- **float, np.ndarray, or pandas.DataFrame**  
  The calculated Gnostic correlation coefficient(s):
  - If both inputs are 1D: returns a float.
  - If either input is 2D: returns a correlation matrix (np.ndarray or pandas DataFrame if input was pandas).

---

## Raises

- **ValueError**
  - If input arrays have different lengths.
  - If inputs are empty or not numpy arrays/pandas Series/DataFrame.
  - If input shapes are incompatible.

---

## Example Usage

```python
import numpy as np
from machinegnostics.metrics import gcorrelation

# Example 1: 1D arrays (robust analog of Pearson correlation)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.9, 2.1, 2.9, 4.2, 4.8])
gcor = gcorrelation(x, y)
print(f"Estimation correlation: {gcor:.3f}")  # Output: Estimation correlation: 0.999

# Example 2: DataFrames (column-wise correlation matrix)
import pandas as pd
df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df2 = pd.DataFrame({'c': [1, 2, 1], 'd': [6, 5, 4]})
corr_matrix = gcorrelation(df1, df2)
print(corr_matrix)
```

---

## Notes

- The location parameter is set by the mean (can be replaced by G-median for higher robustness).
- The geometric mean of the weights is used as the "best" weighting vector.
- For 2D arrays or DataFrames, the function computes the correlation for each pair of columns.
- The output is a DataFrame with appropriate column and index names if the input was pandas.

---

## References
- Kovanic P., Humber M.B (2015) *The Economics of Information - Mathematical Gnostics for Data Analysis*, Chapter 24

---

## License

Machine Gnostics - Machine Gnostics Library  
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.  

---