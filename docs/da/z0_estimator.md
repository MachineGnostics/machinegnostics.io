# Z0Estimator

## Overview

The `Z0Estimator` is a universal estimator for the Z0 point in GDF (Gnostics Distribution Function) distributions. Z0 represents a key location in the distribution:
- For EGDF/ELDF: Z0 is where the PDF reaches its global maximum.
- For QLDF/QGDF: Z0 is the point where the distribution function equals 0.5 (the median or 50th percentile).

This class automatically detects the distribution type and applies advanced estimation strategies to accurately determine Z0, supporting both simple and sophisticated methods.

---

## Key Features

- **Automatic distribution type detection** (EGDF, ELDF, QLDF, QGDF)
- **Multiple estimation strategies**: simple discrete search and advanced optimization (spline, polynomial, interpolation)
- **Robust handling of flat regions and edge cases**
- **Comprehensive diagnostic information**
- **Built-in visualization capabilities**
- **Automatic Z0 assignment to the GDF object**
- **Estimation of Z0 gnostic error properties** (Residual Entropy, RRE)

---

## Parameters

| Parameter      | Type    | Description                                                                                  | Default   |
|----------------|---------|----------------------------------------------------------------------------------------------|-----------|
| gdf_object     | object  | Fitted GDF object (EGDF, ELDF, QLDF, or QGDF). Must be fitted before passing.                | Required  |
| optimize       | bool    | Use advanced optimization methods (spline, polynomial, etc.)                                 | True      |
| verbose        | bool    | Print detailed progress information during estimation                                        | False     |

---

## Attributes

| Attribute         | Type    | Description                                                                                   |
|-------------------|---------|-----------------------------------------------------------------------------------------------|
| gdf               | object  | The fitted GDF object                                                                         |
| gdf_type          | str     | Detected distribution type ('egdf', 'eldf', 'qldf', 'qgdf')                                  |
| optimize          | bool    | Whether advanced optimization is used                                                         |
| verbose           | bool    | Whether verbose output is enabled                                                             |
| find_median       | bool    | True for QLDF/QGDF (find 0.5 point), False for EGDF/ELDF (find PDF max)                      |
| z0                | float   | Estimated Z0 value (None until fit() is called)                                               |
| estimation_info   | dict    | Detailed information about the estimation process                                             |

---

## Methods

| Method                      | Description                                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------------|
| `fit()`                     | Estimates the Z0 point for the given GDF object. Assigns Z0 to the GDF object.              |
| `get_estimation_info()`     | Returns a dictionary with details about the estimation process.                              |
| `plot_z0_analysis()`        | Generates diagnostic plots showing PDF/CDF and the Z0 point.                                |
| `__repr__()`                | Returns a string representation of the estimator and its status.                             |

---

## Example Usage

### 1. EGDF/ELDF (PDF Maximum)

```python
from machinegnostics.magcal import EGDF, Z0Estimator

# Fit your distribution
egdf = EGDF(data=your_data)
egdf.fit()

# Estimate Z0
estimator = Z0Estimator(egdf, verbose=True)
z0 = estimator.fit()
print(f"Z0 at PDF maximum: {z0}")
```

### 2. QLDF/QGDF (Median/0.5 Point)

```python
from machinegnostics.magcal import QLDF, Z0Estimator

# Fit your Q-distribution
qldf = QLDF(data=your_data)
qldf.fit()

# Estimate Z0 at median (0.5)
estimator = Z0Estimator(qldf, optimize=True, verbose=True)
z0 = estimator.fit()
print(f"Z0 at median (0.5): {z0}")
```

### 3. Simple vs Advanced Estimation

```python
# Fast discrete estimation
estimator_simple = Z0Estimator(gdf_object, optimize=False)
z0_simple = estimator_simple.fit()

# Advanced optimization
estimator_advanced = Z0Estimator(gdf_object, optimize=True, verbose=True)
z0_advanced = estimator_advanced.fit()
```

### 4. Getting Diagnostic Information

```python
info = estimator.get_estimation_info()
print(f"Method used: {info['z0_method']}")
print(f"Target type: {info['target_type']}")
print(f"Distribution type: {info['gdf_type']}")
```

### 5. Visualization

```python
estimator.plot_z0_analysis()
# Shows PDF with Z0 point and distribution function/CDF
```

---

## Notes

- The GDF object **must be fitted** before passing to `Z0Estimator`.
- For Q-distributions, Z0 is the point where the distribution function equals 0.5 (median).
- For E-distributions, Z0 is the point where the PDF reaches its global maximum.
- Advanced methods (spline, polynomial, interpolation) are tried in order of reliability.
- The estimated Z0 is **automatically assigned** back to the GDF object.
- Handles flat regions by selecting the middle point.
- Works with any GDF subclass following the standard interface.

---

**Author:** Nirmal Parmar  
**Date:** 2025-09-24

---