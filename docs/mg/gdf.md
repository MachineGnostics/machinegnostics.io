# Gnostic Distribution Functions (GDF)

Gnostic Distribution Functions (GDF) are a new class of probability and density estimators designed for robust, flexible, and assumption-free data analysis. Unlike traditional statistical distributions, GDFs do not require any prior assumptions about the underlying data distribution. Instead, they allow the data to "speak for themselves," making them especially powerful for small, noisy, or uncertain datasets.

## Why Use GDF?

- **No A Priori Assumptions:** GDFs do not rely on predefined parametric forms or statistical models.
- **Robustness:** They are inherently robust to outliers and inner noise, making them suitable for real-world, contaminated, or uncertain data.
- **Flexibility:** GDFs adapt to both homogeneous and heterogeneous data samples, providing detailed insights into data structure.
- **Wide Applicability:** Useful for probability estimation, density estimation, cluster analysis, and homogeneity testing.

## Four Types of Gnostic Distribution Functions

GDFs are organized along two axes:
- **Local vs. Global:** Local functions use weighted irrelevance, while global functions use normalized weights.
- **Estimating vs. Quantifying:** Estimating functions use estimating irrelevance, while quantifying functions use quantifying irrelevance.

This results in four types:
- **ELDF:** Estimating Local Distribution Function (outlier-resistant, highly flexible)
- **EGDF:** Estimating Global Distribution Function (outlier-resistant, unique for each sample)
- **QLDF:** Quantifying Local Distribution Function (inlier-resistant, highly flexible)
- **QGDF:** Quantifying Global Distribution Function (inlier-resistant, unique for each sample)

## Key Concepts

- **Data-Driven:** GDFs are parameterized directly by your data, not by external assumptions.
- **Robustness:** ELDF and EGDF are robust to outliers, while QLDF and QGDF are robust to inliers (dense clusters).
- **Flexibility:** Local functions (ELDF, QLDF) are highly flexible and ideal for cluster analysis and exploring marginal data structures. Global functions (EGDF, QGDF) are best for homogeneity testing and robust probability estimation.
- **Scale Parameter:** The flexibility of local functions is controlled by a scale parameter, allowing you to "zoom in" on data structure.

## Practical Guidance

- Use **local functions** for exploratory, granular analysis and cluster detection.
- Use **global functions** for summary, sample-wide analysis and homogeneity testing.
- Choose **estimating functions** when robustness to outliers is needed.
- Choose **quantifying functions** when robustness to inliers is important.

## Summary

GDFs provide robust, flexible tools for probability and density estimation, especially in challenging data scenarios. The four types allow you to tailor your analysis to the nature of your data and the goals of your study. By removing the constraints of traditional statistical models, GDFs open new possibilities for data-driven insights.

---

For implementation details and examples, see the [Tutorials](../tutorials/).