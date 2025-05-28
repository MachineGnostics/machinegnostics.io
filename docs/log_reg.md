# LogisticRegressor: Robust Logistic Regression with Machine Gnostics

**LogisticRegressor** is a robust, flexible logistic regression model built on the Machine Gnostics framework. It is designed for binary classification tasks and is resilient to outliers, heavy-tailed distributions, and non-Gaussian noise. The model supports polynomial feature expansion, robust weighting, early stopping, and seamless MLflow integration for experiment tracking and deployment.

---

## Overview

The Machine Gnostics LogisticRegressor brings deterministic, event-level modeling to binary classification. It leverages gnostic algebra and geometry to provide robust, interpretable, and reproducible results, even in challenging real-world scenarios.

- **Robust to Outliers:** Gnostic weighting minimizes the influence of noisy or corrupted samples.
- **Polynomial Feature Expansion:** Configurable degree for flexible, nonlinear decision boundaries.
- **Flexible Probability Output:** Choose between gnostic-based or standard sigmoid probabilities.
- **Early Stopping & Convergence:** Monitors loss and entropy for efficient training.
- **MLflow Integration:** For experiment tracking, reproducibility, and deployment.
- **Easy Model Persistence:** Save and load models with joblib.

---

## Key Features

- **Robust to outliers and non-Gaussian noise**
- **Polynomial feature expansion (configurable degree)**
- **Flexible probability output: gnostic or sigmoid**
- **Customizable scaling of data (auto or manual)**
- **Early stopping based on residual entropy or log loss**
- **Full training history tracking (loss, entropy, coefficients, weights)**
- **MLflow integration for model tracking and deployment**
- **Save and load model using joblib**

---

## Parameters

| Parameter          | Type                  | Default   | Description                                                   |
| ------------------ | --------------------- | --------- | ------------------------------------------------------------- |
| `degree`         | int                   | 1         | Degree of the polynomial for feature expansion (1 = linear).  |
| `max_iter`       | int                   | 100       | Maximum number of training iterations.                        |
| `tol`            | float                 | 1e-3      | Convergence threshold for loss or coefficient changes.        |
| `scale`          | {'auto', float}       | 'auto'    | Scaling mode for gnostic transformation.                      |
| `early_stopping` | bool                  | True      | Enables early stopping based on convergence criteria.         |
| `history`        | bool                  | True      | Records training history at each iteration.                   |
| `proba`          | {'gnostic','sigmoid'} | 'gnostic' | Probability output mode.                                      |
| `verbose`        | bool                  | False     | Prints progress and debug information.                        |
| `data_form`      | str                   | 'a'       | Input data form:`'a'` (additive), `'m'` (multiplicative). |

---

## Attributes

- **coefficients**: `ndarray`Final learned polynomial regression coefficients.
- **weights**: `ndarray`Final sample weights after convergence.
- **_history**: `list of dict`
  Training history, including loss, entropy, coefficients, and weights at each iteration.

---

## Methods

### `fit(X, y)`

Fits the model to training data using polynomial expansion and robust loss minimization.

- **X**: array-like, pandas.DataFrame, or numpy.ndarray of shape (n_samples, n_features)Training input samples.
- **y**: array-like or numpy.ndarray of shape (n_samples,)Target binary labels (0 or 1).
- **Returns**:
  `self`: LogisticRegressor (for method chaining)

### `predict(X)`

Predicts class labels (0 or 1) for new input samples using the trained model.

- **X**: array-like, pandas.DataFrame, or numpy.ndarray of shape (n_samples, n_features)Input samples for prediction.
- **Returns**:
  `y_pred`: numpy.ndarray of shape (n_samples,)
  Predicted binary class labels.

### `predict_proba(X)`

Predicts probabilities for new input samples using the trained model.

- **X**: array-like, pandas.DataFrame, or numpy.ndarray of shape (n_samples, n_features)Input samples for probability prediction.
- **Returns**:
  `proba`: numpy.ndarray of shape (n_samples,)
  Predicted probabilities for the positive class (label 1).

### `save_model(path)`

Saves the trained model to disk using joblib.

- **path**: str
  Directory path to save the model.

### `load_model(path)`

Loads a previously saved model from disk.

- **path**: strDirectory path where the model is saved.
- **Returns**:
  Instance of `LogisticRegressor` with loaded parameters.

---

## Example Usage

```python
from machinegnostics.models import LogisticRegressor

# Initialize the model
model = LogisticRegressor(degree=2, proba='gnostic', verbose=True)

# Fit the model
model.fit(X_train, y_train)

# Predict class labels
y_pred = model.predict(X_test)

# Predict probabilities
y_proba = model.predict_proba(X_test)

# Access coefficients and weights
print("Coefficients:", model.coefficients)
print("Weights:", model.weights)

# Save the model
model.save_model("my_logreg_model")

# Load the model
loaded = LogisticRegressor.load_model("my_logreg_model")
y_pred2 = loaded.predict(X_test)
```

---

## Training History

The model records training history at each iteration, accessible via `model._history`.Each entry contains:

- `iteration`: Iteration number
- `loss`: Loss value (gnostic or log loss)
- `entropy`: Residual entropy value
- `coefficients`: Regression coefficients at this iteration
- `weights`: Sample weights at this iteration

This enables detailed analysis and visualization of the training process.


---

## Example

Machine Gnostic Logistic Regression example notebooks: 

- [Example 1](https://github.com/MachineGnostics/machinegnostics.io/blob/main/examples/example_3_moon_data_logreg.ipynb)

- [Example 2](https://github.com/MachineGnostics/machinegnostics.io/blob/main/examples/example_4_logreg_mlflow.ipynb)

![Logistic Regression](./plots/log_reg.png "Logistic Regression")

---

!!! note "Note"

    - The model supports numpy arrays, pandas DataFrames, and pyspark DataFrames as input.
    - For best results, ensure input features are appropriately scaled and encoded.
    - Supports integration with MLflow for experiment tracking and deployment.
    - For more information, visit: [https://machinegnostics.info/](https://machinegnostics.info/)
    - Source code: [https://github.com/MachineGnostics/machinegnostics](https://github.com/MachineGnostics/machinegnostics)

---

## License

Machine Gnostics - Machine Gnostics Library  
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

**Author:** Nirmal Parmar
**Date:** 2025-10-01

---
