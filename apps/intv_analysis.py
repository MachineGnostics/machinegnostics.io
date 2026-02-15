import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from machinegnostics.magcal.gdf.marginal_intv_analysis import IntervalAnalysis


def _parse_numbers(text: str) -> np.ndarray:
    if not text:
        return np.array([])
    parts = [p.strip() for p in text.replace("\n", ",").replace("\t", ",").split(",")]
    vals = []
    for p in parts:
        if p == "":
            continue
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return np.asarray(vals, dtype=float)


def _parse_optional_float(text: str):
    if text is None:
        return None
    t = str(text).strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return t


def _parse_int(text: str, default: int = None):
    try:
        return int(text)
    except Exception:
        return default


def _parse_weights(text: str, n: int) -> np.ndarray | None:
    arr = _parse_numbers(text)
    if arr.size == 0:
        return None
    if arr.size != n:
        st.warning(f"Weights length {arr.size} does not match data length {n}; ignoring weights.")
        return None
    return arr


LEARN_TEXT = (
    "Gnostic Marginal Interval Analysis provides an end-to-end workflow to estimate"
    " meaningful data intervals (tolerance and typical) using GDFs, while checking homogeneity,"
    " enforcing ordering constraints, and providing diagnostics. It fits EGDF (global) and ELDF (local)"
    " models, analyzes Z0 variation as data extends, and derives robust bounds."
)


def main():
    st.set_page_config(page_title="Marginal Interval Analysis | Machine Gnostics", layout="wide")

    st.title("Gnostic Marginal Interval Analysis")
    st.markdown(
        "Estimate robust intervals (tolerance/typical) via EGDF + ELDF with diagnostics."
    )

    # Learn panel
    with st.container(border=True):
        st.subheader("Learn: Interval Analysis")
        tabs = st.tabs(["Overview", "Workflow", "Parameters", "Tips"])
        with tabs[0]:
            st.markdown(
                """
                Gnostic Marginal Interval Analysis estimates meaningful intervals directly from data, without assuming a
                parametric distribution. It balances entropy and information via the Ideal Gnostic Cycle and uses both global
                (EGDF) and local (ELDF) fits to model behavior and bounds.
                """
            )
        with tabs[1]:
            st.markdown(
                """
                1. Fit EGDF for global behavior and homogeneity testing.
                2. If needed, re-fit for non-homogeneous data and adjust bounds.
                3. Fit ELDF for local structure around clusters.
                4. Analyze Z0 (central point) variation as the domain extends.
                5. Estimate tolerance (Z0L, Z0U) and typical (ZL, ZU) intervals.
                6. Provide diagnostics, warnings, and optional visualizations.
                """
            )
        with tabs[2]:
            st.markdown(
                """
                Key parameters:
                - Bounds: DLB/DUB (absolute), LB/UB (probable)
                - Scale S ('auto' or numeric), data_form ('a' additive, 'm' multiplicative)
                - n_points (search), n_points_gdf (smooth curves)
                - Homogeneity check, wedf, weights, optimization settings
                - Dense search fractions, convergence window/threshold, boundary margin
                - Cluster and membership bounds toggles
                """
            )
        with tabs[3]:
            st.markdown(
                """
                Tips:
                - Use EGDF for finding typical bounds and verifying homogeneity.
                - Use ELDF to resolve local structure and cluster-aware bounds.
                - Keep catch=True to enable plotting and diagnostics.
                - Start with S='auto'; adjust only if domain-specific.
                """
            )

    # Data
    st.subheader("Data")
    default_data_str = "-13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10."
    # Initialize session-state backed widget values once
    if "ia_data_text" not in st.session_state:
        st.session_state["ia_data_text"] = default_data_str
    data_text = st.text_area(
        "Enter data points (comma/space/newline separated)",
        height=100,
        placeholder="e.g., 1.2, 3.4, 5.6\n7.8 9.0",
        key="ia_data_text",
    )
    data = _parse_numbers(data_text)
    st.caption(f"Parsed {data.size} points")

    # Options
    st.subheader("Options")
    c1 = st.columns(4)
    with c1[0]:
        DLB = _parse_optional_float(st.text_input("DLB", value=""))
    with c1[1]:
        DUB = _parse_optional_float(st.text_input("DUB", value=""))
    with c1[2]:
        LB = _parse_optional_float(st.text_input("LB", value=""))
    with c1[3]:
        UB = _parse_optional_float(st.text_input("UB", value=""))

    c2 = st.columns(4)
    with c2[0]:
        S_in = st.text_input("S (scale or 'auto')", value="auto")
        S = _parse_optional_float(S_in)
    with c2[1]:
        data_form = st.selectbox("data_form", ["a", "m"], index=0)
    with c2[2]:
        n_points = _parse_int(st.text_input("n_points (search)", value="10"), default=10)
    with c2[3]:
        n_points_gdf = _parse_int(st.text_input("n_points_gdf (smooth)", value="1000"), default=1000)

    c3 = st.columns(4)
    with c3[0]:
        tolerance = _parse_optional_float(st.text_input("tolerance", value="1e-9"))
    with c3[1]:
        homogeneous = st.checkbox("homogeneous", value=True)
    with c3[2]:
        catch = st.checkbox("catch", value=True)
    with c3[3]:
        wedf = st.checkbox("wedf", value=False)

    c4 = st.columns(4)
    with c4[0]:
        z0_optimize = st.checkbox("z0_optimize", value=True)
    with c4[1]:
        opt_method = st.selectbox("opt_method", ["Powell", "SLSQP", "TNC", "CG", "BFGS"], index=0)
    with c4[2]:
        verbose = st.checkbox("verbose", value=False)
    with c4[3]:
        max_data_size = _parse_int(st.text_input("max_data_size", value="1000"), default=1000)

    c5 = st.columns(4)
    with c5[0]:
        flush = st.checkbox("flush", value=True)
    with c5[1]:
        dense_zone_fraction = _parse_optional_float(st.text_input("dense_zone_fraction", value="0.4"))
    with c5[2]:
        dense_points_fraction = _parse_optional_float(st.text_input("dense_points_fraction", value="0.7"))
    with c5[3]:
        boundary_margin_factor = _parse_optional_float(st.text_input("boundary_margin_factor", value="0.001"))

    c6 = st.columns(4)
    with c6[0]:
        convergence_window = _parse_int(st.text_input("convergence_window", value="15"), default=15)
    with c6[1]:
        convergence_threshold = _parse_optional_float(st.text_input("convergence_threshold", value="1e-6"))
    with c6[2]:
        extrema_search_tolerance = _parse_optional_float(st.text_input("extrema_search_tolerance", value="1e-6"))
    with c6[3]:
        min_search_points = _parse_int(st.text_input("min_search_points", value="30"), default=30)

    c7 = st.columns(3)
    with c7[0]:
        gnostic_filter = st.checkbox("gnostic_filter", value=False)
    with c7[1]:
        cluster_bounds = st.checkbox("cluster_bounds", value=True)
    with c7[2]:
        membership_bounds = st.checkbox("membership_bounds", value=True)

    # Weights
    st.subheader("Weights (optional)")
    default_weights_str = ""
    if "ia_weights_text" not in st.session_state:
        st.session_state["ia_weights_text"] = default_weights_str
    weights_text = st.text_input(
        "Weights list (matches data length)",
        key="ia_weights_text",
    )
    weights = _parse_weights(weights_text, data.size)

    # Plot options
    st.subheader("Plot Options")
    plot_GDF = st.checkbox("Plot GDF (ELDF)", value=True)
    plot_intervals = st.checkbox("Plot intervals", value=True)

    # Actions
    buttons = st.columns(3)
    with buttons[0]:
        do_fit = st.button("Fit Analysis", type="primary")
    with buttons[1]:
        do_plot = st.button("Plot")
    with buttons[2]:
        do_reset = st.button("Reset Data & State")

    # Session state for caching
    if "ia_state" not in st.session_state:
        st.session_state["ia_state"] = {}
    if "ia_meta" not in st.session_state:
        st.session_state["ia_meta"] = {}

    # Instantiate
    ia = IntervalAnalysis(
        DLB=DLB,
        DUB=DUB,
        LB=LB,
        UB=UB,
        S=S,
        z0_optimize=z0_optimize,
        tolerance=tolerance if tolerance is not None else 1e-9,
        data_form=data_form,
        n_points=n_points,
        n_points_gdf=n_points_gdf,
        homogeneous=homogeneous,
        catch=catch,
        weights=weights,
        wedf=wedf,
        opt_method=opt_method,
        verbose=verbose,
        max_data_size=max_data_size,
        flush=flush,
        dense_zone_fraction=dense_zone_fraction if dense_zone_fraction is not None else 0.4,
        dense_points_fraction=dense_points_fraction if dense_points_fraction is not None else 0.7,
        convergence_window=convergence_window,
        convergence_threshold=convergence_threshold if convergence_threshold is not None else 1e-6,
        min_search_points=min_search_points,
        boundary_margin_factor=boundary_margin_factor if boundary_margin_factor is not None else 0.001,
        extrema_search_tolerance=extrema_search_tolerance if extrema_search_tolerance is not None else 1e-6,
        gnostic_filter=gnostic_filter,
        cluster_bounds=cluster_bounds,
        membership_bounds=membership_bounds,
    )

    # Auto-clear cached object when data length changes
    prev_meta = st.session_state["ia_meta"].get("IA", {})
    prev_obj = st.session_state["ia_state"].get("IA")
    if prev_obj is not None and prev_meta.get("data_size") is not None and prev_meta["data_size"] != int(data.size):
        st.session_state["ia_state"].pop("IA", None)
        prev_obj = None
    st.session_state["ia_meta"]["IA"] = {"data_size": int(data.size)}
    if prev_obj is not None:
        ia = prev_obj

    # Fit
    if do_fit:
        if data.size < 4:
            st.error("Data must contain at least 4 elements.")
        elif np.any(np.isnan(data)) or np.any(np.isinf(data)):
            st.error("Data contains NaN or Inf.")
        else:
            try:
                ia.fit(data=data, plot=False)
                st.success("Interval analysis completed.")
                st.session_state["ia_state"]["IA"] = ia
                # Show results
                try:
                    res = ia.results()
                    st.subheader("Results")
                    st.json(res)
                except Exception:
                    st.caption("Results retrieval failed; showing object state.")
                    st.write(vars(ia))
            except Exception as e:
                st.error(f"Fit failed: {e}")

    # Reset
    if do_reset:
        # Update session state-backed widget values and clear cached analysis
        st.session_state["ia_data_text"] = default_data_str
        st.session_state["ia_weights_text"] = default_weights_str
        st.session_state.get("ia_state", {}).pop("IA", None)
        st.session_state.get("ia_meta", {}).pop("IA", None)
        st.success("Reset completed. Defaults restored and state cleared.")
        # Rerun to reflect reset values in widgets
        st.rerun()

    # Plot
    if do_plot:
        ia = st.session_state["ia_state"].get("IA", ia)
        if not hasattr(ia, "_fitted") or not getattr(ia, "_fitted"):
            st.error("Analysis not fitted yet. Please click 'Fit Analysis'.")
        else:
            try:
                plt.close("all")
                ia.plot(GDF=plot_GDF, intervals=plot_intervals)
                fig = plt.gcf()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Plot failed: {e}")

    st.markdown("---")
    st.markdown("**Author**: Nirmal Parmar, [Machine Gnostics](https://machinegnostics.info)")
    
if __name__ == "__main__":
    main()
