"""
Machine Gnostics Datasaurus Exploration App
Interactive Streamlit application comparing Machine Gnostics vs Classical Statistics
using the Datasaurus Dozen as a study set.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.metrics import r2_score

from machinegnostics.metrics import (
    correlation as mg_correlation,
    mean as mg_mean,
    median as mg_median,
    robr2,
    root_mean_squared_error as mg_rmse,
)
from machinegnostics.models import LinearRegressor

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Gnostics Datasaurus Exploration",
    page_icon="🦕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #0f1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #e0e6f0 !important;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1a2f52, #0f1a2e);
        border: 1px solid #2d4a7a;
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .kpi-label  { font-size: 12px; color: #8ca0c0; letter-spacing: 0.05em; }
    .kpi-value  { font-size: 28px; font-weight: 700; color: #4fc3f7; margin: 4px 0; }
    .kpi-sub    { font-size: 11px; color: #6a8ab0; }
    .kpi-good   { color: #66bb6a; }
    .kpi-warn   { color: #ffa726; }
    .kpi-diff   { color: #ef5350; }
    .problem-banner {
        background: linear-gradient(90deg, #0d47a1, #1565c0);
        border-left: 5px solid #4fc3f7;
        border-radius: 6px;
        padding: 14px 20px;
        margin-bottom: 20px;
    }
    .problem-banner h3 { color: #e3f2fd; margin: 0 0 6px 0; }
    .problem-banner p  { color: #b3d4f5; margin: 0; font-size: 14px; }
    .step-box {
        background: #1e2d45;
        border: 1px solid #2d4a7a;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #b0c4de;
        margin-bottom: 14px;
    }
    .step-box b { color: #4fc3f7; }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #4fc3f7;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data loading and metric computation
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Datasaurus Dozen...")
def load_datasaurus_dozen_df():
    """Load Datasaurus Dozen from public raw source."""
    urls = [
        "https://raw.githubusercontent.com/jumpingrivers/datasauRus/main/inst/extdata/DatasaurusDozen-Long.tsv",
        "https://raw.githubusercontent.com/jumpingrivers/datasauRus/master/inst/extdata/DatasaurusDozen-Long.tsv",
    ]

    last_error = None
    for url in urls:
        try:
            df = pd.read_csv(url, sep="\t")
            expected_cols = {"dataset", "x", "y"}
            if not expected_cols.issubset(set(df.columns)):
                continue
            df = df[["dataset", "x", "y"]].copy()
            df["x"] = pd.to_numeric(df["x"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df = df.dropna(subset=["x", "y", "dataset"])
            return df
        except Exception as err:  # pragma: no cover
            last_error = err

    raise RuntimeError(
        "Unable to load Datasaurus Dozen from configured sources. "
        f"Last error: {last_error}"
    )


def safe_corr(x, y):
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr) or np.isinf(corr):
        return 0.0
    return float(corr)


def safe_linregress(x, y):
    try:
        slp, intercept, r_val, _, _ = stats.linregress(x, y)
        if np.isnan(r_val):
            r_val = 0.0
        return float(slp), float(intercept), float(r_val)
    except Exception:
        return 0.0, float(np.mean(y)), 0.0


@st.cache_data(show_spinner="Computing metrics for all Datasaurus datasets...")
def compute_all_metrics():
    df = load_datasaurus_dozen_df()
    ds_names = sorted(df["dataset"].unique().tolist())

    metrics_all = {}
    for ds_name in ds_names:
        sub = df[df["dataset"] == ds_name]
        x = sub["x"].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)

        slp, intercept, r_val = safe_linregress(x, y)
        y_pred_cls = intercept + slp * x

        cls = {
            "mean_x": float(np.mean(x)),
            "mean_y": float(np.mean(y)),
            "median_x": float(np.median(x)),
            "median_y": float(np.median(y)),
            "corr": safe_corr(x, y),
            "r2": float(max(0.0, r_val**2)),
            "rmse": float(np.sqrt(np.mean((y - y_pred_cls) ** 2))),
            "slope": float(slp),
            "intercept": float(intercept),
            "y_pred": y_pred_cls,
        }

        model = LinearRegressor(
            max_iter=300,
            early_stopping=True,
            tolerance=1e-6,
            mg_loss="hi",
            history=True,
            verbose=False,
        )
        model.fit(x, y)
        y_pred_mg = np.asarray(model.predict(x), dtype=float)
        weights = getattr(model, "weights", None)

        mg = {
            "mean_x": float(mg_mean(x)),
            "mean_y": float(mg_mean(y)),
            "median_x": float(mg_median(x)),
            "median_y": float(mg_median(y)),
            "corr": float(mg_correlation(x, y)),
            "r2": float(r2_score(y, y_pred_mg)),
            "robr2": float(robr2(y, y_pred_mg, w=weights)),
            "rmse": float(mg_rmse(y, y_pred_mg)),
            "y_pred": y_pred_mg,
            "weights": np.asarray(weights, dtype=float) if weights is not None else None,
            "model": model,
        }

        metrics_all[ds_name] = {"x": x, "y": y, "classical": cls, "mg": mg}

    return metrics_all, ds_names


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def pct_diff(classical_val, mg_val, lower_is_better=False):
    if abs(classical_val) < 1e-12:
        return 0.0, "-", "kpi-sub"
    delta = ((mg_val - classical_val) / abs(classical_val)) * 100
    if lower_is_better:
        delta = -delta
    label = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
    css = "kpi-good" if delta > 0 else ("kpi-warn" if delta > -5 else "kpi-diff")
    return delta, label, css


def kpi_card(label, classical, mg, lower_is_better=False, unit=""):
    _, pct_label, css = pct_diff(classical, mg, lower_is_better)
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{mg:.4f}{unit}</div>
        <div class="kpi-sub">Classical: {classical:.4f}{unit}</div>
        <div class="kpi-sub {css}">MG change: {pct_label}</div>
    </div>
    """


def safe_dataframe(df, **kwargs):
    """Render dataframe with fallback when pyarrow/numpy wheels are incompatible."""
    try:
        st.dataframe(df, **kwargs)
    except Exception as exc:  # pragma: no cover - fallback UI path
        msg = str(exc)
        if "numpy.core.multiarray failed to import" in msg or "pyarrow" in msg:
            st.warning("Falling back to static table view due to pyarrow/numpy compatibility issue.")
            st.table(df)
        else:
            raise


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🥭 Machine Gnostics")
    st.markdown("#### Datasaurus Exploration")
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = "overview"

    st.markdown(
        "<div style='margin-bottom:4px;font-size:12px;color:#8ca0c0;font-weight:bold;'>PRIMARY</div>",
        unsafe_allow_html=True,
    )
    if st.button("▶ Run Exploration", key="nav_overview", use_container_width=True, type="primary"):
        st.session_state.page = "overview"

    st.markdown("---")

    st.markdown(
        "<div style='margin-bottom:12px;font-size:12px;color:#8ca0c0;font-weight:bold;'>EXPLORE DETAILS</div>",
        unsafe_allow_html=True,
    )

    detail_pages = {
        "Why MG Works Better": "why_mg",
        "Mean & Median": "mean_median",
        "Correlation": "correlation",
        "R² & Robust R²": "r2",
        "RMSE": "rmse",
        "Linear Regression": "regression",
        "About": "about",
    }

    for label, key in detail_pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True, type="secondary"):
            st.session_state.page = key

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#445566;text-align:center;'>"
        "Study: Datasaurus Dozen<br>13 Datasets, 5 Metrics</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
try:
    with st.spinner("Loading Datasaurus datasets and computing metrics..."):
        data, DS_NAMES = compute_all_metrics()
except Exception as exc:
    st.error("Could not load Datasaurus Dozen data. Check network access and retry.")
    st.exception(exc)
    st.stop()

page = st.session_state.page


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
if page == "overview":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>🎯 Problem Definition — Same Stats, Different Shapes</h3>
            <p>
            The <b>Datasaurus Dozen</b> contains 13 datasets with near-identical summary statistics
            but radically different visual structures. This exposes a core limitation of classical
            statistics when used alone.<br><br>
            <b>Machine Gnostics</b> applies adaptive information weighting to reveal deeper structural
            differences hidden behind similar means, correlations, and regression scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="step-box">
        <b>👆 How to use this app:</b><br>
        1. Use the <b>left sidebar</b> to navigate sections.<br>
        2. Inspect side-by-side <b>Classical vs Machine Gnostics</b> metrics.<br>
        3. Positive % means MG improves robustness under the metric definition.<br>
        4. In <b>Linear Regression</b>, bubble size and color show MG point weights.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">📋 Datasaurus Dozen — Raw Data Preview</div>', unsafe_allow_html=True)

    tabs = st.tabs([name for name in DS_NAMES])
    for i, ds_name in enumerate(DS_NAMES):
        with tabs[i]:
            df_preview = pd.DataFrame({"X": data[ds_name]["x"], "Y": data[ds_name]["y"]}).round(4)
            col_a, col_b = st.columns([1.2, 1.0])
            with col_a:
                safe_dataframe(df_preview, height=300, use_container_width=True)
            with col_b:
                fig_p, ax_p = plt.subplots(figsize=(3.6, 2.4))
                fig_p.patch.set_facecolor("#0f1a2e")
                ax_p.set_facecolor("#0f1a2e")
                ax_p.scatter(
                    data[ds_name]["x"],
                    data[ds_name]["y"],
                    color="#4fc3f7",
                    s=58,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.4,
                )
                ax_p.set_xlabel("X", color="#8ca0c0")
                ax_p.set_ylabel("Y", color="#8ca0c0")
                ax_p.set_title(ds_name, color="#e0e6f0")
                ax_p.tick_params(colors="#8ca0c0")
                for sp in ax_p.spines.values():
                    sp.set_edgecolor("#2d4a7a")
                st.pyplot(fig_p, use_container_width=False)
                plt.close(fig_p)

    st.markdown('<div class="section-title">📊 Full Study KPI Summary (All Datasets)</div>', unsafe_allow_html=True)

    summary_rows = []
    for ds_name in DS_NAMES:
        cls = data[ds_name]["classical"]
        mg = data[ds_name]["mg"]
        _, p_mean, _ = pct_diff(cls["mean_y"], mg["mean_y"])
        _, p_med, _ = pct_diff(cls["median_y"], mg["median_y"])
        _, p_corr, _ = pct_diff(cls["corr"], mg["corr"])
        _, p_r2, _ = pct_diff(cls["r2"], mg["robr2"])
        _, p_rmse, _ = pct_diff(cls["rmse"], mg["rmse"], lower_is_better=True)

        summary_rows.append(
            {
                "Dataset": ds_name,
                "Mean Δ%": p_mean,
                "Median Δ%": p_med,
                "Corr Δ%": p_corr,
                "R²→RobR² Δ%": p_r2,
                "RMSE Δ% (↓better)": p_rmse,
                "MG RobR²": f"{mg['robr2']:.4f}",
                "CLS R²": f"{cls['r2']:.4f}",
                "MG RMSE": f"{mg['rmse']:.4f}",
                "CLS RMSE": f"{cls['rmse']:.4f}",
            }
        )

    safe_dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Δ% = ((MG - Classical) / |Classical|) × 100. "
        "For RMSE, sign is flipped so positive = MG gives lower error."
    )


# =============================================================================
# PAGE: WHY MG WORKS BETTER
# =============================================================================
elif page == "why_mg":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>Data Truth Revealed: Machine Gnostics vs Classical Statistics</h3>
            <p>
            Datasaurus Dozen shows why plotting is essential: many datasets can share near-identical
            classical statistics while having very different geometry.
            Machine Gnostics emphasizes reliable structure using adaptive point weights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_all = []
    for ds_name in DS_NAMES:
        cls = data[ds_name]["classical"]
        mg = data[ds_name]["mg"]

        drift_corr = abs(mg["corr"] - cls["corr"])
        drift_r2 = abs(mg["robr2"] - cls["r2"])
        drift_rmse = abs(mg["rmse"] - cls["rmse"])

        rows_all.append(
            {
                "Dataset": ds_name,
                "Classical Corr": round(cls["corr"], 4),
                "MG Corr": round(mg["corr"], 4),
                "Classical R²": round(cls["r2"], 4),
                "MG RobR²": round(mg["robr2"], 4),
                "|Corr drift|": round(drift_corr, 4),
                "|R² drift|": round(drift_r2, 4),
                "|RMSE drift|": round(drift_rmse, 4),
                "Interpretation": (
                    "MG indicates meaningful structural re-weighting"
                    if (drift_corr > 0.03 or drift_r2 > 0.03)
                    else "MG and classical are largely aligned"
                ),
            }
        )

    st.markdown("### Summary: What Numbers Reveal")
    safe_dataframe(pd.DataFrame(rows_all), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: MEAN & MEDIAN
# =============================================================================
elif page == "mean_median":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>📊 Mean & Median — Classical vs Machine Gnostics</h3>
            <p>
            Machine Gnostics computes weighted central tendency to reduce sensitivity to
            structurally misleading points.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ds_selected = st.selectbox("Select dataset to inspect:", DS_NAMES)

    cls = data[ds_selected]["classical"]
    mg = data[ds_selected]["mg"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Mean X (MG vs Classical)", cls["mean_x"], mg["mean_x"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Mean Y (MG vs Classical)", cls["mean_y"], mg["mean_y"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Median X (MG vs Classical)", cls["median_x"], mg["median_x"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Median Y (MG vs Classical)", cls["median_y"], mg["median_y"]), unsafe_allow_html=True)

    rows = []
    for ds_name in DS_NAMES:
        c = data[ds_name]["classical"]
        m = data[ds_name]["mg"]
        _, pm_y, _ = pct_diff(c["mean_y"], m["mean_y"])
        _, pmed_y, _ = pct_diff(c["median_y"], m["median_y"])
        rows.append(
            {
                "Dataset": ds_name,
                "CLS Mean Y": round(c["mean_y"], 4),
                "MG Mean Y": round(m["mean_y"], 4),
                "Mean Y Δ%": pm_y,
                "CLS Median Y": round(c["median_y"], 4),
                "MG Median Y": round(m["median_y"], 4),
                "Median Y Δ%": pmed_y,
            }
        )
    safe_dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: CORRELATION
# =============================================================================
elif page == "correlation":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>🔗 Correlation — Classical Pearson vs Machine Gnostics</h3>
            <p>
            Classical Pearson correlation can hide structural differences. MG correlation is adaptive,
            emphasizing stable association rather than leverage-driven artifacts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_corr = []
    for ds_name in DS_NAMES:
        c = data[ds_name]["classical"]
        m = data[ds_name]["mg"]
        _, pct, _ = pct_diff(c["corr"], m["corr"])
        rows_corr.append(
            {
                "Dataset": ds_name,
                "Classical Pearson r": round(c["corr"], 6),
                "MG Correlation": round(m["corr"], 6),
                "Δ% (MG vs Classical)": pct,
            }
        )
    safe_dataframe(pd.DataFrame(rows_corr), use_container_width=True, hide_index=True)

    fig_corr, ax_corr = plt.subplots(figsize=(12, 4.8))
    fig_corr.patch.set_facecolor("#0f1a2e")
    ax_corr.set_facecolor("#0f1a2e")

    cls_corrs = [data[d]["classical"]["corr"] for d in DS_NAMES]
    mg_corrs = [data[d]["mg"]["corr"] for d in DS_NAMES]
    x_pos = np.arange(len(DS_NAMES))

    ax_corr.bar(x_pos - 0.2, cls_corrs, 0.35, label="Classical", color="#66bb6a", alpha=0.85)
    ax_corr.bar(x_pos + 0.2, mg_corrs, 0.35, label="Machine Gnostics", color="#4fc3f7", alpha=0.85)
    ax_corr.set_xticks(x_pos)
    ax_corr.set_xticklabels(DS_NAMES, rotation=35, ha="right", color="#8ca0c0")
    ax_corr.set_ylabel("Correlation", color="#8ca0c0")
    ax_corr.set_title("Correlation: Classical vs Machine Gnostics", color="#e0e6f0")
    ax_corr.legend(facecolor="#0f1a2e", labelcolor="#e0e6f0")
    ax_corr.tick_params(colors="#8ca0c0")
    for sp in ax_corr.spines.values():
        sp.set_edgecolor("#2d4a7a")
    st.pyplot(fig_corr, use_container_width=True)
    plt.close(fig_corr)


# =============================================================================
# PAGE: R² & ROBUST R²
# =============================================================================
elif page == "r2":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>📈 R² vs Robust R² (RobR²)</h3>
            <p>
            Classical R² can be overly optimistic when leverage points dominate fit quality.
            MG Robust R² penalizes such behavior and reflects broader structural fit quality.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_r2 = []
    for ds_name in DS_NAMES:
        c = data[ds_name]["classical"]
        m = data[ds_name]["mg"]
        _, pct, _ = pct_diff(c["r2"], m["robr2"])
        rows_r2.append(
            {
                "Dataset": ds_name,
                "Classical R²": round(c["r2"], 6),
                "MG RobR²": round(m["robr2"], 6),
                "MG R² (std)": round(m["r2"], 6),
                "RobR² Δ%": pct,
            }
        )
    safe_dataframe(pd.DataFrame(rows_r2), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: RMSE
# =============================================================================
elif page == "rmse":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>📉 RMSE — Root Mean Squared Error</h3>
            <p>
            Positive RMSE Δ% indicates MG achieves lower error after robust weighting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_rmse = []
    for ds_name in DS_NAMES:
        c = data[ds_name]["classical"]
        m = data[ds_name]["mg"]
        _, pct, _ = pct_diff(c["rmse"], m["rmse"], lower_is_better=True)
        rows_rmse.append(
            {
                "Dataset": ds_name,
                "Classical RMSE": round(c["rmse"], 6),
                "MG RMSE": round(m["rmse"], 6),
                "RMSE Δ% (↓better→+%)": pct,
                "Verdict": "✅ MG lower error" if m["rmse"] <= c["rmse"] else "ℹ️ Classical lower raw RMSE",
            }
        )
    safe_dataframe(pd.DataFrame(rows_rmse), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: LINEAR REGRESSION
# =============================================================================
elif page == "regression":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>📐 Linear Regression — Classical OLS vs Machine Gnostics</h3>
            <p>
            Bubble color and size encode MG weights: large green points are trusted more,
            small red points are down-weighted.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ds_sel = st.selectbox("Select dataset for detailed view:", DS_NAMES, key="reg_ds")

    c = data[ds_sel]["classical"]
    m = data[ds_sel]["mg"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("Classical R²", 1.0, c["r2"]), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("MG Robust R²", 1.0, m["robr2"]), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("RMSE (lower=better)", c["rmse"], m["rmse"], lower_is_better=True), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("Correlation", c["corr"], m["corr"]), unsafe_allow_html=True)

    x_arr = data[ds_sel]["x"]
    y_arr = data[ds_sel]["y"]
    y_pred_cls = c["y_pred"]
    y_pred_mg = m["y_pred"]
    weights = m["weights"]

    fig_reg, ax_reg = plt.subplots(figsize=(9, 5))
    fig_reg.patch.set_facecolor("#0f1a2e")
    ax_reg.set_facecolor("#0f1a2e")

    if weights is not None and weights.size == len(x_arr):
        w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        sizes = 60 + w_norm * 220
        sc = ax_reg.scatter(
            x_arr,
            y_arr,
            s=sizes,
            c=weights,
            cmap="RdYlGn",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.4,
            label="Data (size & color = MG weight)",
            zorder=3,
        )
        cbar = fig_reg.colorbar(sc, ax=ax_reg)
        cbar.set_label("MG Weight", color="#8ca0c0")
        cbar.ax.yaxis.set_tick_params(color="#8ca0c0")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8ca0c0")
    else:
        ax_reg.scatter(x_arr, y_arr, s=80, color="#4fc3f7", alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)

    idx = np.argsort(x_arr)
    ax_reg.plot(
        x_arr[idx],
        y_pred_cls[idx],
        "--",
        color="#66bb6a",
        lw=2.4,
        label=f"Classical OLS  R²={c['r2']:.4f}  RMSE={c['rmse']:.4f}",
    )
    ax_reg.plot(
        x_arr[idx],
        y_pred_mg[idx],
        "-",
        color="#ef5350",
        lw=2.4,
        label=f"Machine Gnostics  RobR²={m['robr2']:.4f}  RMSE={m['rmse']:.4f}",
    )

    ax_reg.set_xlabel("X", color="#8ca0c0")
    ax_reg.set_ylabel("Y", color="#8ca0c0")
    ax_reg.set_title(f"{ds_sel} — Classical OLS vs Machine Gnostics", color="#e0e6f0")
    ax_reg.legend(fontsize=9, facecolor="#0f1a2e", labelcolor="#e0e6f0", loc="best")
    ax_reg.tick_params(colors="#8ca0c0")
    for sp in ax_reg.spines.values():
        sp.set_edgecolor("#2d4a7a")
    st.pyplot(fig_reg, use_container_width=True)
    plt.close(fig_reg)

    st.markdown("#### Full Regression KPI Table — All Datasets")
    reg_rows = []
    for ds_name in DS_NAMES:
        c2 = data[ds_name]["classical"]
        m2 = data[ds_name]["mg"]
        _, pr2, _ = pct_diff(c2["r2"], m2["robr2"])
        _, prmse, _ = pct_diff(c2["rmse"], m2["rmse"], lower_is_better=True)
        _, pcorr, _ = pct_diff(c2["corr"], m2["corr"])
        reg_rows.append(
            {
                "Dataset": ds_name,
                "CLS R²": round(c2["r2"], 4),
                "MG RobR²": round(m2["robr2"], 4),
                "RobR² Δ%": pr2,
                "CLS RMSE": round(c2["rmse"], 4),
                "MG RMSE": round(m2["rmse"], 4),
                "RMSE Δ% (↓better→+%)": prmse,
                "CLS Corr": round(c2["corr"], 4),
                "MG Corr": round(m2["corr"], 4),
                "Corr Δ%": pcorr,
                "CLS Slope": round(c2["slope"], 4),
                "CLS Intercept": round(c2["intercept"], 4),
            }
        )
    safe_dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: ABOUT
# =============================================================================
elif page == "about":
    st.markdown('<div class="section-title">ℹ️ About this Datasaurus Exploration App</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### What is this app?
        This Streamlit app studies **Machine Gnostics** against classical statistics
        on the **Datasaurus Dozen** collection.

        ### Why Datasaurus?
        Datasaurus Dozen demonstrates that many datasets can share similar summary statistics
        (mean, variance, correlation) while having very different visual patterns.

        ### What is compared?
        - Mean and median
        - Correlation
        - R² and robust R² (RobR²)
        - RMSE
        - Regression fit behavior under adaptive MG weighting

        ### Data source
        - jumpingrivers/datasauRus
        - File: DatasaurusDozen-Long.tsv
        """
    )
