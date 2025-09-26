# Gnostic Tests

This tutorial covers advanced data diagnostics in the Machine Gnostics framework:

- **Data Homogeneity Test**: Analyze if your data sample is homogeneous using EGDF and PDF analysis.
- **Data Scedasticity Test**: Diagnose homoscedasticity or heteroscedasticity using gnostic variance and regression.
- **Data Membership Test**: Check if a value can be considered a member of a homogeneous sample.

---

## Data Homogeneity Test

Analyze data homogeneity for EGDF objects using probability density function analysis.

The homogeneity criterion is based on the mathematical properties and expected PDF behavior of EGDF according to gnostic theory principles. Homogeneous data should produce a distribution with a single density maximum, while non-homogeneous data will exhibit multiple maxima or negative density values.

!!! example "Homogeneity Test with Outlier"
	```python
	import numpy as np
	from machinegnostics.magcal import DataHomogeneity, EGDF

	# Example: homogeneous data with an outlier
	data = np.array([-3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	# Example: non-homogeneous data
	data = np.array([-13.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	egdf = EGDF(verbose=False)
	egdf.fit(data=data)

	dh = DataHomogeneity(gdf=egdf, verbose=False, flush=True)
	is_homogeneous = dh.fit(plot=True)
	print("Is the data homogeneous? ", is_homogeneous)
	dh.results()
	```

**Output:**
```text
Is the data homogeneous?  False
{'has_negative_pdf': np.True_,
 'num_maxima': 1,
 'extrema_type': 'maxima',
 'gdf_type': 'egdf',
 'is_homogeneous': False,
 ...}
```

---

## Data Scedasticity Test

Gnostic Scedasticity Test for Homoscedasticity and Heteroscedasticity

This class provides a method to check for homoscedasticity and heteroscedasticity in data, inspired by fundamental principles rather than standard statistical tests. It uses gnostic variance and gnostic linear regression, which are based on the Machine Gnostics framework.

!!! example "Scedasticity Test"
	```python
	import numpy as np
	from machinegnostics.magcal import DataScedasticity

	X = np.array([0., 0.4, 0.8, 1.2, 1.6, 2. ])
	y = np.array([17.89408548, 69.61586934, -7.19890572, 9.37670866, -10.55673099, 16.57855348])

	ds = DataScedasticity(verbose=False)
	is_homoscedastic = ds.fit(x=X, y=y)
	print("Is the data homoscedastic? ", is_homoscedastic)
	```

**Output:**
```text
Is the data homoscedastic?  False
```

---

## Data Membership Test

Test whether a value can be considered a member of a homogeneous data sample using the EGDF framework.

!!! example "Data Membership Test"
	```python
	import numpy as np
	from machinegnostics.magcal import DataMembership, EGDF

	data = np.array([-3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	egdf = EGDF(verbose=False)
	egdf.fit(data=data)

	dm = DataMembership(egdf=egdf, verbose=False)
	lsb, usb = dm.fit()
	print(f"Lower and Upper Sample Bounds to check Data Membership: LSB: {lsb}, USB: {usb}")
	```

**Output:**
```text
Lower and Upper Sample Bounds to check Data Membership: LSB: -3, USB: 10.013
```

---

## Tips

- All diagnostic classes (`DataHomogeneity`, `DataScedasticity`, `DataMembership`) follow a similar API: create an object, fit your data, and inspect results.
- Use the `plot=True` option in `.fit()` for visual diagnostics where available.
- For advanced usage and parameter tuning, see the [API Reference](../da/homogeneity.md), [scedasticity](../da/scedasticity.md), and [membership](../da/membership.md).

---

**Next:**
Explore more tutorials and real-world examples in the [Examples](examples.md) section!
