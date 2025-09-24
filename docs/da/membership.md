# DataMembership: Gnostic Membership Test (Machine Gnostics)

The `DataMembership` class provides a robust method to test whether a value can be considered a member of a homogeneous data sample, using the EGDF (Estimating Global Distribution Function) framework. It determines the bounds within which new data points can be added to a sample without disrupting its homogeneity.

---

## Overview

DataMembership answers the question: "Will the homogeneous sample remain homogeneous after extension by a new value?" The class uses EGDF and DataHomogeneity to check the effect of adding candidate values to the sample, and computes the Lower Sample Bound (LSB) and Upper Sample Bound (USB) for membership.

- **EGDF-Based:** Only works with EGDF objects.
- **Homogeneity-Driven:** Membership is defined by preservation of sample homogeneity.
- **Adaptive Bound Search:** Finds minimum and maximum values that keep the sample homogeneous.
- **Diagnostic:** Tracks errors, warnings, and search parameters.
- **Visualization:** Plots EGDF, PDF, and membership bounds.

---

## Key Features

- **Tests membership of candidate values in homogeneous samples**
- **Computes LSB and USB for membership range**
- **Integrates with EGDF and DataHomogeneity**
- **Adaptive, iterative bound search**
- **Detailed logging and error tracking**
- **Visualization of membership bounds and EGDF**

---

## Parameters

| Parameter             | Type                  | Default   | Description                                                      |
|-----------------------|-----------------------|-----------|------------------------------------------------------------------|
| `egdf`                | EGDF                  | required  | Fitted EGDF object (must be fitted and contain data)             |
| `verbose`             | bool                  | True      | Print detailed logs and diagnostics                              |
| `catch`               | bool                  | True      | Store errors, warnings, and results                              |
| `tolerance`           | float                 | 1e-3      | Tolerance for numerical calculations                             |
| `max_iterations`      | int                   | 100       | Maximum iterations for bound search                              |
| `initial_step_factor` | float                 | 0.001     | Initial step size factor for adaptive search                     |

---

## Attributes

- **LSB**: `float or None`  
  Lower Sample Bound (minimum value that keeps sample homogeneous)
- **USB**: `float or None`  
  Upper Sample Bound (maximum value that keeps sample homogeneous)
- **is_homogeneous**: `bool`  
  Indicates whether the original sample is homogeneous
- **params**: `dict`  
  Stores results, errors, warnings, and search parameters
- **fitted**: `bool`  
  Indicates whether membership analysis has been completed

---

## Methods

### `fit()`

Performs membership analysis to determine LSB and USB.

**Returns:**  
`Tuple[float or None, float or None]` — The calculated LSB and USB values. Returns None for a bound if it cannot be determined.

---

### `plot(plot_smooth=True, plot='both', bounds=True, figsize=(12, 8))`

Generates a plot of the EGDF and PDF with membership bounds and other relevant information.

- **plot_smooth**: `bool`  
  Plot smoothed EGDF and PDF
- **plot**: `str`  
  'gdf', 'pdf', or 'both'
- **bounds**: `bool`  
  Include data bounds in the plot
- **figsize**: `tuple`  
  Figure size

**Returns:**  
None (displays plot)

---

### `results()`

Returns the analysis results stored in the `params` attribute.

**Returns:**  
`dict` — Contains LSB, USB, is_homogeneous, search parameters, errors, and warnings.

---

## Example Usage

```python
import numpy as np
from machinegnostics.magcal import EGDF, DataMembership
data = np.array([ -13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

egdf_instance = EGDF(data)
egdf_instance.fit()

membership = DataMembership(egdf_instance, verbose=True)
lsb, usb = membership.fit()

print(f"Lower Bound: {lsb}, Upper Bound: {usb}")

membership.plot()

results = membership.results()
print(results)
```

---

## Notes

- Only works with EGDF objects; sample must be homogeneous for membership analysis.
- LSB and USB define the range of values that can be added to the sample without losing homogeneity.
- Errors and warnings are tracked in the results dictionary.
- Visualization shows EGDF, PDF, and membership bounds for interpretability.

---

**Author:** Nirmal Parmar  
**Date:** 2025-09-24

---