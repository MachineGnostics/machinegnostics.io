# DataCluster: Advanced Cluster Boundary Detection for Gnostic Distribution Functions (Machine Gnostics)

The `DataCluster` class identifies main cluster boundaries (LCB and UCB) from probability density functions of Gnostic Distribution Functions (GDFs): ELDF, EGDF, QLDF, and QGDF. It uses normalized PDF analysis, derivative-based methods, and shape detection algorithms for robust cluster identification.

---

## Overview

DataCluster provides automated, robust cluster boundary detection for GDFs. It adapts its algorithm based on the type of GDF, using derivative thresholds, valley detection, and slope analysis to find the main cluster region. The class is designed for scientific, engineering, and data science applications where interpretable cluster boundaries are needed.

- **Supports All GDF Types:** ELDF, EGDF, QLDF, QGDF.
- **PDF Normalization:** Ensures consistent analysis across distributions.
- **Shape Detection:** W-shape/U-shape/heterogeneous detection for QLDF.
- **Derivative-Based Boundaries:** Uses first and second derivatives for boundary detection.
- **Fallback Strategies:** Falls back to data bounds if boundary detection fails.
- **Diagnostic:** Tracks errors, warnings, and method details.
- **Visualization:** Plots PDF, boundaries, and derivative analysis.

---

## Key Features

- **Automated cluster boundary detection for GDFs**
- **PDF normalization and robust derivative analysis**
- **Shape-based valley detection for QLDF**
- **Adaptive thresholding and slope analysis**
- **Comprehensive error handling and diagnostics**
- **Visualization of PDF, boundaries, and cluster regions**

---

## Parameters

| Parameter                | Type                | Default  | Description                                     |
| ------------------------ | ------------------- | -------- | ----------------------------------------------- |
| `gdf`                  | ELDF/EGDF/QLDF/QGDF | required | Fitted GDF object with `pdf_points` available |
| `verbose`              | bool                | False    | Print detailed logs and diagnostics             |
| `catch`                | bool                | True     | Store errors, warnings, and results             |
| `derivative_threshold` | float               | 0.01     | Threshold for ELDF/EGDF boundary detection      |
| `slope_percentile`     | int                 | 70       | Percentile for QLDF/QGDF slope-based detection  |

---

## Attributes

- **LCB**: `float or None`Cluster Lower Boundary (left boundary of main cluster)
- **UCB**: `float or None`Cluster Upper Boundary (right boundary of main cluster)
- **z0**: `float or None`Characteristic point of the distribution
- **S_opt**: `float or None`Optimal scale parameter from GDF
- **pdf_normalized**: `ndarray or None`Min-max normalized PDF values [0,1]
- **pdf_original**: `ndarray or None`Original PDF values
- **params**: `dict`Complete analysis results, boundaries, diagnostics, and method details
- **fitted**: `bool`
  Indicates whether clustering analysis has been completed

---

## Methods

### `fit(plot=False)`

Performs cluster boundary detection analysis.

- **plot**: `bool` (optional)
  If True, generates a plot of the PDF, detected boundaries, and derivative analysis.

**Returns:**
`Tuple[float or None, float or None]` — The detected LCB and UCB values. Returns None for a bound if it cannot be determined.

---

### `results()`

Returns a comprehensive cluster analysis results dictionary.

**Returns:**
`dict` — Contains LCB, UCB, cluster width, GDF type, Z0, S_opt, method details, errors, and warnings.

---

### `plot(figsize=(12, 8))`

Creates a visualization of the PDF, detected boundaries, and derivative analysis.

- **figsize**: `tuple` (default: (12, 8))
  Figure size

**Returns:**
None (displays plot)

---

## Example Usage

```python
import numpy as np
from machinegnostics.magcal import ELDF, DataCluster

data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

eldf = ELDF()
eldf.fit(data)

cluster = DataCluster(gdf=eldf, verbose=True)
CLB, CUB = cluster.fit(plot=True)

results = cluster.results()
print(f"Lower boundary: {results['LCB']}")
print(f"Upper boundary: {results['UCB']}")
print(f"Cluster width: {results['cluster_width']}")
```

---

## Notes

- Clustering works best with local distribution functions (ELDF, QLDF).
- Global functions (EGDF, QGDF) have limited clustering effectiveness due to uniqueness constraints.
- QLDF W-shape detection is effective for central clusters between outlying regions.
- For heterogeneous data with multiple clusters, consider splitting the dataset before analysis.
- Errors and warnings are tracked in the results dictionary.

---

## References

- Gnostic Distribution Function theory and cluster analysis methods (see mathematical gnostics literature).

---

**Author:** Nirmal Parmar   
**Date:** 2025-09-24

---
