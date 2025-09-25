# Installation Guide

Machine Gnostics is distributed as a standard Python package and is designed for easy installation and integration into your data science workflow. The library has been tested on macOS with Python 3.11 and is fully compatible with standard data science libraries.

---

## 1. Create a Python Virtual Environment

It is best practice to use a virtual environment to manage your project dependencies and avoid conflicts with other Python packages.

### macOS & Linux
```bash
# Create a new virtual environment named 'mg-env'
python3 -m venv mg-env
# Activate the environment
source mg-env/bin/activate
```

### Windows
```cmd
# Create a new virtual environment named 'mg-env'
python -m venv mg-env
# Activate the environment
mg-env\Scripts\activate
```

---

## 2. Install Machine Gnostics

Install the Machine Gnostics library using pip:

### macOS & Linux
```bash
pip install machinegnostics
```

### Windows
```cmd
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
```

You can also check the installation with pip:

### macOS & Linux
```bash
pip show machinegnostics
```

### Windows
```cmd
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

- **Activate Your Environment:**
  Always activate your virtual environment before installing or running Machine Gnostics.
  
  **macOS & Linux:**
  ```bash
  source mg-env/bin/activate
  # or for conda
  conda activate myenv
  ```
  **Windows:**
  ```cmd
  mg-env\Scripts\activate
  # or for conda
  conda activate myenv
  ```

- **Check Your Python Version:**
  Ensure you are using Python 3.8 or newer.
  
  **macOS & Linux:**
  ```bash
  python3 --version
  ```
  **Windows:**
  ```cmd
  python --version
  ```

- **Upgrade pip:**
  An outdated pip can cause installation errors. Upgrade pip before installing:
  
  **macOS & Linux:**
  ```bash
  pip install --upgrade pip
  ```
  **Windows:**
  ```cmd
  pip install --upgrade pip
  ```

- **Install from a Clean Environment:**  
  If you encounter conflicts, try creating a fresh virtual environment and reinstalling.

- **Check Your Internet Connection:**  
  Download errors often result from network issues. Make sure you are connected.

- **Permission Issues:**  
  If you see permission errors, avoid using `sudo pip install`. Instead, use a virtual environment.

- **Still Stuck?**  
  - Double-check the [installation instructions](installation.md).
  - [Contact us](contact.md) or open an issue on [GitHub](https://github.com/MachineGnostics/machinegnostics).

---

Machine Gnostics is designed for simplicity and reliability, making robust machine learning accessible for all Python users.
