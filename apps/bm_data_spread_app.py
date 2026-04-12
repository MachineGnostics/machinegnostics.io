import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from machinegnostics.magcal import EGDF, ELDF, DataIntervals
from scipy.stats import t

# Data
water_g = np.array([
    9.96037, 9.96454, 9.9632, 9.97152, 9.9683,
    9.97413, 9.96806, 9.96165, 9.96397, 9.9544
])

water_ml = np.array([
    9.98734, 9.99152, 9.99017, 9.99852, 9.99529,
    10.00113, 9.99505, 9.98862, 9.99095, 9.98135
])

st.title("[Benchmark] - Mean and Data Spread Comparison Dashboard")

# Sidebar for data selection and setup
st.sidebar.header("Data & Setup")
dataset = st.sidebar.selectbox("Select Dataset", ["Water Mass (g)", "Water Volume (ml)"], index=0)
use_custom = st.sidebar.checkbox("Use custom data instead")

if use_custom:
    st.sidebar.write("**Note:** Custom data works best with small datasets (less than 20 values) for stable benchmark comparisons.")
    custom_data_input = st.sidebar.text_area("Paste your data as comma-separated values (e.g., 1.2, 3.4, 5.6)", "")
    if custom_data_input:
        try:
            data = np.array([float(x.strip()) for x in custom_data_input.split(',') if x.strip()])
            unit = st.sidebar.text_input("Unit", "units")
            title_suffix = "Custom Data"
            st.sidebar.success(f"Loaded {len(data)} values.")
        except ValueError:
            st.sidebar.error("Invalid input. Please enter numbers separated by commas.")
            data = water_g if dataset == "Water Mass (g)" else water_ml
            unit = "g" if dataset == "Water Mass (g)" else "ml"
            title_suffix = "Mass" if dataset == "Water Mass (g)" else "Volume"
    else:
        data = water_g if dataset == "Water Mass (g)" else water_ml
        unit = "g" if dataset == "Water Mass (g)" else "ml"
        title_suffix = "Mass" if dataset == "Water Mass (g)" else "Volume"
else:
    if dataset == "Water Mass (g)":
        data = water_g
        unit = "g"
        title_suffix = "Mass"
    else:
        data = water_ml
        unit = "ml"
        title_suffix = "Volume"

# Display selected data in sidebar
st.sidebar.subheader("Selected Data")
st.sidebar.write(f"**{title_suffix} Measurements:**")
st.sidebar.write(data.tolist())
st.sidebar.write(f"**Number of measurements:** {len(data)}")
if not use_custom:
    st.sidebar.write("**Data Source:** [here](https://sisu.ut.ee/measurement/41-naidisulesandeks/)")

# Confidence Interval Setup in sidebar
st.sidebar.subheader("Confidence Interval Setup")
confidence_level = st.sidebar.slider("Confidence Level (%)", min_value=80, max_value=99, value=95, step=1)
confidence_decimal = confidence_level / 100
n = len(data)
calculated_t_value = t.ppf((1 + confidence_decimal) / 2, df=n-1)

t_value_option = st.sidebar.radio("t-value Option", ["Auto", "Custom"])
if t_value_option == "Custom":
    t_value = st.sidebar.number_input("t-value", min_value=1.0, max_value=10.0, value=float(calculated_t_value), step=0.01)
    st.sidebar.write(f"Calculated t-value: {calculated_t_value:.3f}")
else:
    t_value = calculated_t_value

st.sidebar.write(f"Using t-value: {t_value:.3f}")

# Main content
st.markdown("""
This app gives an easy, high-level comparison of benchmark ranges from different methods: Statistical vs Mathematical Gnostics.
Use it to quickly see how each method estimates the center and spread of your measurements.
""")

# Add custom CSS for red button
st.markdown("""
<style>
.stButton > button {
    background-color: #FF0000;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
}
.stButton > button:hover {
    background-color: #FF0000;
    color: white;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
### What this dashboard helps you do
- Compare benchmark results side by side in a clear visual format.
- Understand whether ranges are narrow, wide, or shifted between methods.
- Communicate outcomes to mixed audiences without deep statistical language.

Machine Gnostics models are designed to be flexible and assumption-free, so they can be useful when real data does not follow statistical assumptions.
""")

# Explanations in expanders
with st.expander("Concept Overview"):
    st.markdown("""
This comparison brings together two styles of benchmarking:

- Statistical method: builds a confidence-based range around the average.
- Mathematical Gnostics method: builds ranges based on how values are distributed in the data.

From the Machine Gnostics documentation, EGDF and ELDF belong to the
Gnostic Distribution Function (GDF) family, which provides non-parametric,
distribution-focused analysis.

- EGDF (Estimating Global Distribution Function): emphasizes the overall distribution shape.
- ELDF (Estimating Local Distribution Function): emphasizes local structure and regional variation.

The goal is not to pick a winner in every case, but to help you understand how method choice can change benchmark results.
                
**Gnostic vs. Statistical Interval Analysis:**
    Gnostic interval analysis does not rely on probabilistic or statistical assumptions. Instead, it uses algebraic and geometric properties of the data and distribution functions, providing deterministic, reproducible, and interpretable intervals even for small, noisy, or non-Gaussian datasets. This is fundamentally different from classical statistical interval estimation, which depends on distributional assumptions and sampling theory.
""")

with st.expander("How to Read the Benchmark"):
    st.markdown("""
Focus on three simple parts:

- Center: the main reference value for each method.
- Range: the lower-to-upper interval shown in plots and table.
- Width: how broad each interval is.

If one method gives a wider range, it is expressing more uncertainty. If it gives a narrower range, it is expressing tighter confidence in the estimate.

Tip: when data includes outliers or non-normal behavior, data-driven methods can reveal patterns that a purely symmetric statistical range may miss.
""")

# Run Benchmark Button in sidebar
if st.sidebar.button("Run Benchmark Analysis"):
    # Computations
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    expanded_uncertainty = t_value * (std_val / np.sqrt(n))

    # EGDF
    egdf = EGDF()
    egdf.fit(data, plot=False)
    ia_egdf = DataIntervals(egdf)
    ia_egdf.fit(plot=False)
    egdf_res = ia_egdf.results()

    # ELDF for comparison
    eldf = ELDF()
    eldf.fit(data, plot=False)
    ia_eldf = DataIntervals(eldf)
    ia_eldf.fit(plot=False)
    eldf_res = ia_eldf.results()

    # Visualizations
    st.header("Benchmark Results")

    # Bar Plot: Mean Value Comparison
    st.subheader("Mean Value Comparison: Statistical vs EGDF vs ELDF")
    methods = ['Statistical', 'EGDF', 'ELDF']
    means = [mean_val, egdf_res['Z0'], eldf_res['Z0']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, means, color=['#1976D2', '#388E3C', '#7B1FA2'])  # Professional blue, green, purple
    ax.set_title(f'Mean Values Comparison ({unit})')
    ax.set_ylabel(f'Mean ({unit})')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{mean:.5f}', ha='center', va='bottom')

    st.pyplot(fig)

    # Data Scatter Plot
    st.subheader("Data Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(len(data)), data, alpha=0.7, label='Data points', color='#424242')  # Dark gray
    ax.axhline(y=mean_val, color='#1976D2', linestyle='--', label=f'Statistical Mean: {mean_val:.5f} {unit}')
    ax.fill_between(range(len(data)), mean_val - expanded_uncertainty, mean_val + expanded_uncertainty, alpha=0.3, color='#1976D2', label=f'Statistical CI: ±{expanded_uncertainty:.5f} {unit}')
    ax.axhline(y=egdf_res['Z0'], color='#388E3C', linestyle='--', label=f'EGDF Mean: {egdf_res["Z0"]:.5f} {unit}')
    ax.fill_between(range(len(data)), egdf_res['Z0L'], egdf_res['Z0U'], alpha=0.3, color='#388E3C', label=f'EGDF Tolerance')
    ax.fill_between(range(len(data)), egdf_res['ZL'], egdf_res['ZU'], alpha=0.2, color='#66BB6A', label=f'EGDF Typical')
    ax.set_xlabel('Measurement Index')
    ax.set_ylabel(f'{title_suffix} ({unit})')
    ax.set_title(f'Benchmark Comparison: Statistical vs EGDF - {title_suffix}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Error Bar Comparison
    st.subheader("Error Bar Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = [0, 1, 2]
    methods_short = ['Statistical', 'EGDF', 'ELDF']
    stat_err = expanded_uncertainty
    eldf_mean = eldf_res['Z0']
    eldf_tol_err = [[eldf_mean - eldf_res['Z0L']], [eldf_res['Z0U'] - eldf_mean]]
    eldf_typ_err = [[eldf_mean - eldf_res['ZL']], [eldf_res['ZU'] - eldf_mean]]

    ax.errorbar(x_pos[0], mean_val, yerr=stat_err, fmt='o', capsize=5, label='Statistical', color='#1976D2')
    ax.errorbar(x_pos[1], egdf_res['Z0'], yerr=[[egdf_res['Z0'] - egdf_res['Z0L']], [egdf_res['Z0U'] - egdf_res['Z0']]], fmt='o', capsize=5, label='EGDF Tolerance', color='#388E3C')
    ax.errorbar(x_pos[1], egdf_res['Z0'], yerr=[[egdf_res['Z0'] - egdf_res['ZL']], [egdf_res['ZU'] - egdf_res['Z0']]], fmt='o', capsize=5, label='EGDF Typical', color='#66BB6A', linewidth=1)
    ax.errorbar(x_pos[2], eldf_mean, yerr=eldf_tol_err, fmt='o', capsize=5, label='ELDF Tolerance', color='#7B1FA2', linewidth=2)
    ax.errorbar(x_pos[2], eldf_mean, yerr=eldf_typ_err, fmt='o', capsize=5, label='ELDF Typical', color='#BA68C8', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods_short)
    ax.set_ylabel(f'{title_suffix} ({unit})')
    ax.set_title('Error Bars Comparison: Statistical vs EGDF vs ELDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Metrics (range-based table for direct comparison)
    st.subheader("Benchmark Metrics")
    stat_lower = mean_val - expanded_uncertainty
    stat_upper = mean_val + expanded_uncertainty

    metrics_rows = [
        {
            "Method": "Statistical",
            "Center": f"{mean_val:.5f}",
            "Lower": f"{stat_lower:.5f}",
            "Upper": f"{stat_upper:.5f}",
            "Range": f"{stat_lower:.5f} to {stat_upper:.5f} {unit}",
            "Width": f"{(stat_upper - stat_lower):.5f}",
        },
        {
            "Method": "EGDF (Tolerance)",
            "Center": f"{egdf_res['Z0']:.5f}",
            "Lower": f"{egdf_res['Z0L']:.5f}",
            "Upper": f"{egdf_res['Z0U']:.5f}",
            "Range": f"{egdf_res['Z0L']:.5f} to {egdf_res['Z0U']:.5f} {unit}",
            "Width": f"{(egdf_res['Z0U'] - egdf_res['Z0L']):.5f}",
        },
        {
            "Method": "EGDF (Typical)",
            "Center": f"{egdf_res['Z0']:.5f}",
            "Lower": f"{egdf_res['ZL']:.5f}",
            "Upper": f"{egdf_res['ZU']:.5f}",
            "Range": f"{egdf_res['ZL']:.5f} to {egdf_res['ZU']:.5f} {unit}",
            "Width": f"{(egdf_res['ZU'] - egdf_res['ZL']):.5f}",
        },
        {
            "Method": "ELDF (Tolerance)",
            "Center": f"{eldf_res['Z0']:.5f}",
            "Lower": f"{eldf_res['Z0L']:.5f}",
            "Upper": f"{eldf_res['Z0U']:.5f}",
            "Range": f"{eldf_res['Z0L']:.5f} to {eldf_res['Z0U']:.5f} {unit}",
            "Width": f"{(eldf_res['Z0U'] - eldf_res['Z0L']):.5f}",
        },
        {
            "Method": "ELDF (Typical)",
            "Center": f"{eldf_res['Z0']:.5f}",
            "Lower": f"{eldf_res['ZL']:.5f}",
            "Upper": f"{eldf_res['ZU']:.5f}",
            "Range": f"{eldf_res['ZL']:.5f} to {eldf_res['ZU']:.5f} {unit}",
            "Width": f"{(eldf_res['ZU'] - eldf_res['ZL']):.5f}",
        },
    ]

    st.table(metrics_rows)


# learn more and links
st.header("Learn More")
st.markdown("""
- **Machine Gnostics Distribution Functions (GDFs)** are a model family for robust, distribution-aware analysis.
- They are described as flexible and non-parametric, meaning they do not rely on strict normal-distribution assumptions.
- **EGDF** (global view) and **ELDF** (local view) are both part of this GDF family.
- The broader Machine Gnostics framework also emphasizes diagnostic analysis, such as interval, variance, and consistency checks.

- Documentation links:
  - [Models Overview](https://docs.machinegnostics.com/latest/da/da_models/)
  - [EGDF](https://docs.machinegnostics.com/latest/da/egdf/)
  - [ELDF](https://docs.machinegnostics.com/latest/da/eldf/)
""")