# DataIntervals: Robust Interval Estimation Engine (Machine Gnostics)

The `DataIntervals` class provides robust, adaptive, and diagnostic interval estimation for Gnostic Distribution Function (GDF) classes such as ELDF, EGDF, QLDF, and QGDF. It estimates meaningful data intervals (tolerance, typical intervals) based on the behavior of the GDF's central parameter (Z0) as the data is extended, enforcing ordering constraints and providing detailed diagnostics.

---

## Overview

DataIntervals is designed for advanced interval analysis in scientific, engineering, and reliability applications. It adaptively scans the data domain, recomputes the GDF as needed, and extracts intervals that reflect the true structure of the data. The class enforces natural ordering constraints and provides comprehensive diagnostics and visualization.

- **Adaptive Search:** Dense scanning near Z0, sparse near boundaries for efficiency and accuracy.
- **Robustness:** Optional GDF recomputation and gnostic filtering for outlier resistance.
- **Diagnostics:** Tracks warnings, errors, and parameter settings.
- **Ordering Constraint:** Ensures intervals satisfy ZL < Z0L < Z0 < Z0U < ZU.
- **Visualization:** Plots Z0 variation, intervals, and data coverage.

---

## Key Features

- **Adaptive interval scanning and estimation**
- **Robust to outliers and noise**
- **Supports ELDF, EGDF, QLDF, QGDF**
- **Ordering constraint enforcement**
- **Detailed diagnostics and error tracking**
- **Visualization of Z0 variation and intervals**
- **Memory-efficient with optional flushing**

---

## Parameters

| Parameter                    | Type                | Default  | Description                                          |
| ---------------------------- | ------------------- | -------- | ---------------------------------------------------- |
| `gdf`                      | ELDF/EGDF/QLDF/QGDF | required | Fitted GDF object to analyze                         |
| `n_points`                 | int                 | 100      | Number of search points for interval estimation      |
| `dense_zone_fraction`      | float               | 0.4      | Fraction of domain near Z0 for dense search          |
| `dense_points_fraction`    | float               | 0.7      | Fraction of points in dense zone                     |
| `convergence_window`       | int                 | 15       | Window size for convergence detection                |
| `convergence_threshold`    | float               | 1e-6     | Threshold for Z0 convergence                         |
| `min_search_points`        | int                 | 30       | Minimum search points before checking convergence    |
| `boundary_margin_factor`   | float               | 0.001    | Margin to avoid searching at boundaries              |
| `extrema_search_tolerance` | float               | 1e-6     | Tolerance for detecting extrema in Z0 variation      |
| `gdf_recompute`            | bool                | False    | Recompute GDF for each candidate datum               |
| `gnostic_filter`           | bool                | False    | Apply gnostic clustering to filter outlier Z0 values |
| `catch`                    | bool                | True     | Store warnings/errors internally                     |
| `verbose`                  | bool                | False    | Print detailed progress and diagnostics              |
| `flush`                    | bool                | False    | Flush memory after fitting to save resources         |

---

## Attributes

- **ZL**: `float`Lower bound of the typical data interval
- **Z0L**: `float`Lower bound of the tolerance interval (Z0-based)
- **Z0**: `float`Central value (Z0) of the original GDF
- **Z0U**: `float`Upper bound of the tolerance interval (Z0-based)
- **ZU**: `float`Upper bound of the typical data interval
- **tolerance_interval**: `float`Width of the tolerance interval (Z0U - Z0L)
- **typical_data_interval**: `float`Width of the typical data interval (ZU - ZL)
- **ordering_valid**: `bool`Whether the ordering constraint (ZL < Z0L < Z0 < Z0U < ZU) is satisfied
- **params**: `dict`Dictionary of parameters, warnings, errors, and results
- **search_results**: `dict`
  Raw search results for datum values and corresponding Z0s

---

## Methods

### `fit(plot=False)`

Runs the interval estimation process. Optionally plots results.

- **plot**: `bool` (optional)
  If True, automatically plot the interval analysis results after fitting.

**Returns:**
None (results stored in attributes and params)

---

### `results()`

Returns a dictionary of estimated interval results and bounds.

**Returns:**
`dict` — Contains keys such as `'LB', 'LSB', 'DLB', 'LCB', 'LSD', 'ZL', 'Z0L', 'Z0', 'Z0U', 'ZU', 'USD', 'UCB', 'DUB', 'USB', 'UB'`

    -**LB**: Lower Bound
    The practical lower limit for the interval (may be set by user or inferred).

    -**LSB**: Lower Sample (Membership) Bound
    The lowest value for which data is homogeneous.

    -**DLB**: Data Lower Bound
    The absolute minimum value present in the data.

    -**LCB**: Lower Cluster Bound
    The lower edge of the main data cluster.

    -**LSD**: Lower Standard Deviation Bound
    The lowest value as per gnostic standard deviation.

    -**ZL**: Z0 Lower Interval
    The lower bound of the typical interval.

    -**Z0L**: Z0 Lower Bound
    The lower bound of the tolerance interval.

    -**Z0**: Central Value (Gnostic Mean)
    The central parameter of the distribution (gnostic mean).

    -**Z0U**: Z0 Upper Bound
    The upper bound of the tolerance interval.

    -**ZU**: Z0 Upper Interval
    The upper bound of the typical interval.

    -**USD**: Upper Support/Domain Bound
    The highest value in the support or domain of the fitted distribution.

    -**UCB**: Upper Cluster Bound
    The upper edge of the main data cluster.

    -**DUB**: Data Upper Bound
    The absolute maximum value present in the data.

    -**USB**: Upper Sample (Membership) Bound
    The highest value for which data is homogeneous (membership analysis).

    -**UB**: Upper Bound
    The practical upper limit for the interval (may be set by user or inferred).

---

---

### `plot_intervals(figsize=(12, 8))`

Plots the Z0 variation and estimated intervals.

- **figsize**: `tuple` (default: (12, 8))
  Size of the matplotlib figure

**Returns:**
None (displays plot)

---

### `plot(figsize=(12, 8))`

Plots the GDF, PDF, and intervals on the data domain.

- **figsize**: `tuple` (default: (12, 8))
  Size of the matplotlib figure

**Returns:**
None (displays plot)

---

## Example Usage

```python
import numpy as np
from machinegnostics.magcal import ELDF, DataIntervals

data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

# ELDF
eld = ELDF()
eld.fit(data)

di = DataIntervals(eld, n_points=200, gdf_recompute=True, gnostic_filter=True, verbose=True)
di.fit()

print(di.results())
di.plot_intervals()
di.plot()
```

---

## Notes

- For best results, use with ELDF or QLDF and set 'wedf=False' in the GDF.
- Increasing 'n_points' improves accuracy but increases computation time.
- Enable 'gdf_recompute' and 'gnostic_filter' for maximum robustness, especially with noisy data.
- The class is designed for research and diagnostic use; adjust parameters for your data and application.
- Ordering constraint is enforced for interval validity; warnings are issued if violated.
- All warnings and errors are stored in the `params` attribute for later inspection.

---

**Author:** Nirmal Parmar   
**Date:** 2025-09-24

---
