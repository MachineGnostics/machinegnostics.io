# make_regression_check_data: Synthetic Regression Dataset

The `make_regression_check_data` function generates a synthetic linear regression dataset. It creates a simple linear relationship with additive Gaussian noise ($y = mx + c + \epsilon$), serving as a reliable "hello world" test for validating regression models within the Machine Gnostics framework.

---

## Overview

This utility simplifies the creation of regression datasets for testing:
-   **Equation**: $y = \text{slope} \times X + \text{intercept} + \text{noise}$
-   **Purpose**: Unit testing models, verifying pipeline integrity, and demonstrating basic regression capabilities (e.g., `GnosticLinearRegression`).
-   **Reproducibility**: Uses a fixed seed (default 42) to ensure consistent results across runs.

---

## Parameters

| Parameter     | Type   | Description                                                                 | Default |
| :------------ | :----- | :-------------------------------------------------------------------------- | :------ |
| `n_samples`   | int    | The number of data points to generate.                                      | `20`    |
| `slope`       | float  | The true coefficient (slope) of the linear relationship.                    | `3.5`   |
| `intercept`   | float  | The true intercept (bias) of the linear relationship.                       | `10.0`  |
| `noise_level` | float  | Standard deviation of Gaussian noise added to target.                         | `2.0`   |
| `seed`        | int    | Random seed for reproducibility.                                            | `42`    |

---

## Returns

| Return | Type          | Description                                         |
| :----- | :------------ | :-------------------------------------------------- |
| `X`    | numpy.ndarray | Input feature array of shape `(n_samples,)`. Values uniform in [0, 10]. |
| `y`    | numpy.ndarray | Target array of shape `(n_samples,)`.                 |

---

## Example Usage

```python
from machinegnostics.datasets import make_regression_check_data
import numpy as np

# Generate regression data with 50 samples
X, y = make_regression_check_data(n_samples=50)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
# Output:
# X shape: (50,)
# y shape: (50,)
```

---

**Author:** Nirmal Parmar
