
# Usage Guide

This guide provides a comprehensive overview of how to use the **Machine Gnostics** library for robust data analysis and machine learning based on Machine Gnostics principles. Machine Gnostics offers robust regression models, gnostic metrics, and alternative statistical tools designed to be resilient to outliers and corrupted data.

---

## 1. Importing Machine Gnostics

After installation, you can import Machine Gnostics and its modules in your Python scripts or notebooks:

```python
import machinegnostics as mg
from machinegnostics.models import RobustRegressor
from machinegnostics.metrics import robr2, gmmfe, divI, evalMet, hc
from machinegnostics.magcal import gmedian, gvariance, gautocovariance, gcorrelation, gcovariance
```

---

## 2. Robust Regression

Machine Gnostics provides robust regression models that are less sensitive to outliers.

**Example: Using `RobustRegressor`**

```python
from machinegnostics.models import RobustRegressor

# X: feature matrix, y: target vector
model = RobustRegressor()
model.fit(X, y)
y_pred = model.predict(X_test)
```

---

## 3. Gnostic Metrics

Evaluate your models with robust, gnostic metrics:

```python
from machinegnostics.metrics import robr2, gmmfe

score = robr2(y_true, y_pred)
gmmfe = gmmfe(y_true, y_pred)
hc = hc(y_true, y_pred, case='i') # estimating case
```

Other available metrics:

- `divI`: Divergence Index
- `evalMet`: General evaluation metric
- `hc`: Relavance of the given data samples

## 4. Gnostic Statistical Tools

Machine Gnostics includes robust alternatives to classical statistics:

```python
gmed = gmedian(data)
gmod = gmodulus(data)
gvar = gvariance(data)
gacov = gautocovariance(data1, data2)
gcor = gcorrelation(data1, data2)
gcov = gcovariance(data1, data2)
```

---

## 5. Example Workflow

```python
import numpy as np
from machinegnostics.models import RobustRegressor
from machinegnostics.metrics import robr2

# Generate synthetic data
X = np.random.randn(100, 3)
y = 2 * X[:, 0] - X[:, 1] + np.random.randn(100) * 0.5

# Fit robust regression
model = RobustRegressor()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluate
score = robr2(y, y_pred)
print("Robust R2:", score)

```

---

## 6. Troubleshooting

- **ImportError**: Ensure Machine Gnostics is installed and your `PYTHONPATH` includes the `src` directory.
- **Unexpected Results**: Check for outliers or corrupted data in your input.

---

For further help, open an issue on [GitHub](https://github.com/MachineGnostics) or contact the maintainers.
