# DataHomogeneity: Homogeneity Analysis for EGDF (Machine Gnostics)

The `DataHomogeneity` class provides robust, numerical homogeneity analysis for Estimating Global Distribution Functions (EGDF) by examining the shape and characteristics of their probability density functions (PDF). It is designed to detect outliers, clusters, and non-homogeneous structure in data using gnostic theory principles.

---

## Overview

DataHomogeneity analyzes the fitted EGDF's PDF to determine if the underlying data is homogeneous. Homogeneity is defined by the presence of a single global maximum (unimodal PDF) and the absence of negative density values. The class uses robust peak detection, configurable smoothing, and comprehensive diagnostics to provide reliable results.

**Gnostic vs. Statistical Homogeneity:**  
Gnostic homogeneity analysis is based on the algebraic and geometric properties of the data and the EGDF, not on statistical or probabilistic assumptions. It is deterministic, reproducible, and sensitive to both outliers and clusters, making it fundamentally different from classical statistical homogeneity tests.

- **Assumption-Free:** No parametric or probabilistic assumptions.
- **Numerical:** Decisions are made based on numerical analysis, not visual inspection.
- **Robust:** Detects outliers and clusters via PDF maxima.
- **Diagnostic:** Tracks errors, warnings, and analysis parameters.
- **Memory-Efficient:** Optional flushing of large arrays after analysis.
- **Visualization:** Built-in plotting for PDF and detected maxima.

---

## Key Features

- **Automatic EGDF validation and homogeneity testing**
- **Robust peak detection with configurable smoothing**
- **Comprehensive error and warning tracking**
- **Memory management with optional data flushing**
- **Detailed visualization of analysis results**
- **Integration with EGDF parameter systems**

---

## Parameters

| Parameter           | Type                  | Default   | Description                                                      |
|---------------------|-----------------------|-----------|------------------------------------------------------------------|
| `gdf`               | EGDF                  | required  | Fitted EGDF object (must have catch=True and be fitted)          |
| `verbose`           | bool                  | True      | Print detailed progress, warnings, and results                   |
| `catch`             | bool                  | True      | Store all analysis results and metadata                          |
| `flush`             | bool                  | False     | Clear large arrays after analysis to save memory                 |
| `smoothing_sigma`   | float                 | 1.0       | Gaussian smoothing parameter for PDF preprocessing               |
| `min_height_ratio`  | float                 | 0.01      | Minimum relative height threshold for peak detection             |
| `min_distance`      | int or None           | None      | Minimum separation between detected peaks (auto if None)         |

---

## Attributes

- **is_homogeneous**: `bool or None`  
  Primary analysis result (None before fit, True/False after analysis)
- **picks**: `List[Dict]`  
  Detected maxima with detailed information (index, position, value, global flag)
- **z0**: `float or None`  
  Global optimum value from EGDF or detected from PDF
- **global_extremum_idx**: `int or None`  
  Array index of the global maximum
- **fitted**: `bool`  
  Indicates if analysis has been completed

---

## Methods

### `fit(plot=False)`

Performs comprehensive homogeneity analysis on the EGDF object.

- **plot**: `bool` (optional)  
  If True, generates plots for visual inspection of the analysis results

**Returns:**  
`bool` — True if data is homogeneous, False otherwise

---

### `results()`

Retrieves comprehensive homogeneity analysis results and metadata.

**Returns:**  
`dict` — Contains keys such as `'is_homogeneous', 'picks', 'z0', 'global_extremum_idx', 'analysis_parameters', 'gdf_parameters', 'errors', 'warnings'`

---

### `plot(figsize=(12, 8), title=None)`

Visualizes the PDF, detected maxima, and homogeneity status.

- **figsize**: `tuple` (default: (12, 8))  
  Figure size in inches
- **title**: `str or None`  
  Custom plot title

**Returns:**  
None (displays plot)

---

## Example Usage

```python
import numpy as np
from machinegnostics.magcal import EGDF, DataHomogeneity

# Homogeneous data
data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
egdf = EGDF(data=data, catch=True)
egdf.fit()

# Homogeneity analysis
homogeneity = DataHomogeneity(egdf, verbose=True)
is_homogeneous = homogeneity.fit(plot=True)
print(f"Data is homogeneous: {is_homogeneous}")

# Access results
results = homogeneity.results()
print(f"Number of maxima detected: {len(results['picks'])}")
```

---

## Notes

- Only supports EGDF objects (not QGDF, ELDF, or QLDF)
- Homogeneity is defined by a single global maximum and no negative PDF values
- Outliers and clusters are detected as additional maxima
- Numerical analysis is preferred over visual inspection for reliability
- Use `flush=True` for large datasets to save memory
- All errors and warnings are tracked in the results dictionary

---

**Author:** Nirmal Parmar  
**Date:** 2025-09-24

---