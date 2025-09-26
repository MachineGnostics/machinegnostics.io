# Machine Gnostics Measures

!!! note
    **Why Machine Gnostics?**  
    Unlike classical statistics, which rely on probabilistic averages, Machine Gnostics computes measures like mean, median, standard deviation, and variance using irrelevance and fidelity from gnostic theory. This approach is assumption-free and robust to outliers, revealing the true diagnostic properties of your data.

---

## 1. Sample Data

Let’s start with a small dataset that includes an outlier, to see how Machine Gnostics handles challenging real-world data.

!!! example "Sample Data"
    ```python
    import numpy as np

    # Example data with an outlier
    data = np.array([-13.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print("Data:", data)
    ```

---

## 2. Gnostic Mean

The gnostic mean is robust to outliers and does not assume a specific data distribution.

!!! example "Gnostic Mean"
    ```python
    import machinegnostics as mg

    mean = mg.mean(data)
    print("Gnostic Mean:", mean)
    ```

---

## 3. Gnostic Median

The gnostic median provides a robust central value, even for small or skewed samples.

!!! example "Gnostic Median"
    ```python
    import machinegnostics as mg

    median = mg.median(data)
    print("Gnostic Median:", median)
    ```

---

## 4. Gnostic Standard Deviation

Unlike classical standard deviation, the gnostic version returns a lower and upper bound, reflecting uncertainty more realistically.

!!! example "Gnostic Standard Deviation"
    ```python
    import machinegnostics as mg

    std_dev_lb, std_dev_ub = mg.std(data)
    print("Gnostic Std Dev (lower, upper):", std_dev_lb, std_dev_ub)
    ```

---

## 5. Gnostic Variance

Gnostic variance is always between 0 and 1, as it is calculated using irrelevance rather than squared deviations.

!!! example "Gnostic Variance"
    ```python
    import machinegnostics as mg

    var = mg.variance(data)
    print("Gnostic Variance:", var)
    ```

---

## Tips

- **Robustness:** Try changing or adding more outliers to your data and see how the gnostic measures respond compared to classical statistics.
- **Integration:** All functions follow standard Python/NumPy conventions and can be used in data science workflows.
- **Documentation:** See the [API Reference](../metrics/g_mean.md) for advanced options and parameter tuning.

---

**Next:**  
Explore more tutorials and real-world examples in the [Examples](examples.md) section!