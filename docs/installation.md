# Installation Guide

Machine Gnostics is distributed as a standard Python package and is designed for easy installation and integration into your data science workflow. The library has been tested on macOS with Python 3.11 and is fully compatible with standard data science libraries.

---

## 1. Create a Python Virtual Environment

It is best practice to use a virtual environment to manage your project dependencies and avoid conflicts with other Python packages.

```bash
# Create a new virtual environment named 'mag-env'
python3 -m venv mg-env

# Activate the environment (macOS/Linux)
source mg-env/bin/activate

# (On Windows, use: machinegnostics-env\Scripts\activate)
```

---

## 2. Install Machine Gnostics

Install the Machine Gnostics library using pip:

```bash
# main installation
pip install machinegnostics
```

This command will install Machine Gnostics and automatically resolve its dependencies.


---

## 3. Verify Installation

You can verify that Machine Gnostics and its dependencies are installed correctly by importing them in a Python session:

```python

# check import
import machinegnostics

print("imported successfully!")

# check with pip
pip show machinegnostics
```

---

## 4. Quick Usage Example

Machine Gnostics is designed to be as simple to use as other machine learning libraries. You can call its functions and classes directly after installation.

```python
import numpy as np
from machinegnostics.models.regression import LinearRegressor

# Example data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# Create and fit a robust polynomial regression model
model = LinearRegressor()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

print("Predictions:", y_pred)
```

---

## 5. Platform and Environment

- **Operating System:** Tested on macOS and Windows 11
- **Python Version:** 3.11 recommended
- **Dependencies:** Compatible with NumPy, pandas, SciPy, and other standard data science libraries

---

## 6. Troubleshooting

- Ensure your virtual environment is activated before installing or running Machine Gnostics.
- If you encounter issues, try upgrading pip:

  ```bash
  pip install --upgrade pip
  ```
!!! note "Help"
    - For further help, [consult](contact) us or open an issue on the [GitHub repository](https://github.com/MachineGnostics/machinegnostics).

---

Machine Gnostics is designed for simplicity and reliability, making robust machine learning accessible for all Python users.
