import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import sys
import logging
import contextlib

# GDF classes
from machinegnostics.magcal import ELDF, EGDF, QLDF, QGDF


def _parse_numbers(text: str) -> np.ndarray:
    """Parse a comma/space/newline-separated string into a 1D numpy array of floats."""
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
            # Ignore non-parsable tokens gracefully
            pass
    return np.asarray(vals, dtype=float)


def _parse_optional_float(text: str):
    """Return float if parseable, otherwise return original text (e.g., 'auto')."""
    if text is None:
        return None
    t = str(text).strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return t  # pass through strings like 'auto'


def _parse_int(text: str, default: int = None):
    try:
        return int(text)
    except Exception:
        return default


def _parse_weights(text: str, n: int) -> np.ndarray | None:
    """Parse weights list; return None if empty or mismatched length."""
    arr = _parse_numbers(text)
    if arr.size == 0:
        return None
    if arr.size != n:
        st.warning(f"Weights length {arr.size} does not match data length {n}; ignoring weights.")
        return None
    return arr


LEARN_TEXT = {
    "ELDF": (
        "Estimating Local Distribution Function (ELDF) focuses on local behavior around data, "
        "supporting additive/multiplicative forms, Z0 (Gnostic Mean) estimation, bounds, and variable scale `S`. "
        "Use it for detailed local analysis, peak detection, and local density estimation."
    ),
    "EGDF": (
        "Estimating Global Distribution Function (EGDF) provides a global representation of the distribution, "
        "with automatic parameter estimation, optional bounds, and robust optimization. Useful for risk analysis "
        "and process-wide modeling."
    ),
    "QLDF": (
        "Quantifying Local Distribution Function (QLDF) characterizes local distribution features around critical points, "
        "including automatic Z0 identification and variable scale options. Ideal for learning local minima and neighborhood behavior."
    ),
    "QGDF": (
        "Quantifying Global Distribution Function (QGDF) analyzes global characteristics with configurable bounds, scale, and optimization. "
        "Use it to quantify global distribution features across datasets."
    ),
}


def main():
    st.set_page_config(page_title="GDF Explorer | Machine Gnostics", layout="wide")

    st.title("Gnostic Distribution Functions (GDF) Explorer")
    st.markdown(
        "Explore and learn four GDFs — ELDF, EGDF, QLDF, and QGDF. "
        "Paste your data below, tune parameters, fit models, and visualize results."
    )

    # Learning panel (top, not at bottom)
    with st.container(border=True):
        st.subheader("Learn: What are GDFs?")
        tabs = st.tabs(["Overview", "Axes of Selection", "Class Guide", "Parameters", "Notes"])

        with tabs[0]:
            st.markdown(
                """
                Gnostic Distribution Functions (GDF) are a new class of probability and density estimators designed for robust,
                flexible, and assumption-free data analysis. Unlike traditional statistical distributions, GDFs do not require any
                prior assumptions about the underlying data distribution. Instead, they allow the data to "speak for themselves,"
                making them especially powerful for small, noisy, or uncertain datasets.
                
                GDFs model the probability of individual data points based on the Ideal Gnostic Cycle, balancing entropy (uncertainty)
                and information.
                """
            )

        with tabs[1]:
            st.markdown(
                """
                ### The Two Axes of Selection

                - **Estimating (E) vs. Quantifying (Q):**
                  - **E-Type (Estimating):** Robust to outliers; maximizes information by focusing on the "central" data (the truth) and suppressing extreme values.
                    Use when you want to filter out noise or find the "normal" behavior.
                  - **Q-Type (Quantifying):** Robust to inliers; emphasizes peripheral data and suppresses the central "noise".
                    Use when the "signal" is an anomaly (e.g., detecting a rare defect or a faint astronomical signal).

                - **Local (L) vs. Global (G):**
                  - **L-Type (Local):** Highly flexible; can model multiple peaks (multimodality). Use to see detailed inner structure, such as identifying multiple clusters.
                  - **G-Type (Global):** More rigid; generally unimodal. Use to model overall trend, test for homogeneity, or establish valid data bounds.
                """
            )

        with tabs[2]:
            st.markdown(
                """
                ### Class Descriptions & Use Cases

                | Class | Type | Interpretation & Best Use |
                | :--- | :--- | :--- |
                | **EGDF** | Estimating / Global | The "Standard" Model. Fits a smooth, single-peak curve that ignores outliers. Use for: Homogeneity testing, estimating "normal" data bounds ($LSB, USB$), and general probability estimation. |
                | **ELDF** | Estimating / Local | The Structure Seeker. Wraps tightly around data clusters while dampening outliers. Use for: Marginal cluster analysis and interval analysis. |
                | **QGDF** | Quantifying / Global | The Anomaly Modeler. Fits an overall curve that highlights extreme values. Use for: Detecting deviations when the "normal" center is considered noise. |
                | **QLDF** | Quantifying / Local | The Detail Seeker. Flexible fitting that emphasizes peripheral details. Use for: Detailed analysis of complex data structures where outliers are the target. |
                """
            )

        with tabs[3]:
            st.markdown(
                """
                ### Interpreting Parameters

                - **Scale Parameter ($S$):** Controls the resolution/curvature. Small $S$ reveals fine details (narrow kernels), large $S$ smoothes the distribution.
                  In Global functions (EGDF/QGDF), $S$ is typically optimized automatically to find the best fit.
                - **Irrelevance ($h$):** A measure of error or uncertainty of a datum. GDFs are essentially weighted averages of these irrelevances.
                """
            )

        with tabs[4]:
            st.markdown(
                """
                ### Notes

                - If you want to find the "true" value and ignore errors, use **EGDF**.
                - If you want to find distinct groups within the data, use **ELDF**.
                """
            )

    # Select GDF type
    gdf_choice = st.radio("Select Distribution Function", ["ELDF", "EGDF", "QLDF", "QGDF"], horizontal=True)
    st.info(LEARN_TEXT[gdf_choice])

    # Data input (no file upload; typed only)
    st.subheader("Data")
    default_data_str = "-13.5, 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10."
    data_text = st.text_area(
        "Enter data points (comma/space/newline separated)",
        value=st.session_state.get("data_text", default_data_str),
        height=100,
        placeholder="e.g., 1.2, 3.4, 5.6\n7.8 9.0",
        key="data_text",
    )
    data = _parse_numbers(data_text)
    st.caption(f"Parsed {data.size} points")

    # Common options across classes
    st.subheader("Options")

    cols1 = st.columns(4)
    with cols1[0]:
        DLB = _parse_optional_float(st.text_input("DLB (Data Lower Bound)", value=""))
    with cols1[1]:
        DUB = _parse_optional_float(st.text_input("DUB (Data Upper Bound)", value=""))
    with cols1[2]:
        LB = _parse_optional_float(st.text_input("LB (Lower Probable Bound)", value=""))
    with cols1[3]:
        UB = _parse_optional_float(st.text_input("UB (Upper Probable Bound)", value=""))

    cols2 = st.columns(4)
    with cols2[0]:
        S_in = st.text_input("S (scale or 'auto')", value="auto")
        S = _parse_optional_float(S_in)
    with cols2[1]:
        data_form = st.selectbox("Data Form", options=["a", "m"], index=0, help="'a' additive, 'm' multiplicative")
    with cols2[2]:
        n_points = _parse_int(st.text_input("n_points", value="500"), default=500)
    with cols2[3]:
        tolerance = _parse_optional_float(st.text_input("tolerance", value="1e-9"))

    cols3 = st.columns(4)
    with cols3[0]:
        homogeneous = st.checkbox("homogeneous", value=True)
    with cols3[1]:
        catch = st.checkbox("catch (store intermediates)", value=True)
    with cols3[2]:
        wedf = st.checkbox("wedf (weighted EDF)", value=False)
    with cols3[3]:
        verbose = st.checkbox("verbose", value=False)

    cols4 = st.columns(4)
    with cols4[0]:
        opt_method = st.selectbox("opt_method", ["Powell", "SLSQP", "TNC", "CG", "BFGS"], index=0)
    with cols4[1]:
        max_data_size = _parse_int(st.text_input("max_data_size", value="1000"), default=1000)
    with cols4[2]:
        flush = st.checkbox("flush (free large arrays)", value=True)
    with cols4[3]:
        z0_optimize = st.checkbox("z0_optimize", value=True)

    # Class-specific options
    varS = None
    minimum_varS = None
    if gdf_choice in ("ELDF", "QLDF"):
        cols5 = st.columns(2)
        with cols5[0]:
            varS = st.checkbox("varS (variable scale)", value=False)
        with cols5[1]:
            minimum_varS = _parse_optional_float(st.text_input("minimum_varS", value="0.1"))

    # Weights (optional)
    st.subheader("Weights (optional)")
    default_weights_str = ""
    weights_text = st.text_input(
        "Weights list (matches data length)",
        value=st.session_state.get("weights_text", default_weights_str),
        key="weights_text",
    )
    weights = _parse_weights(weights_text, data.size)

    # Plot options
    st.subheader("Plot Options")
    plot_smooth = st.checkbox("Smooth curve", value=True)
    bounds = st.checkbox("Show bounds", value=True)
    extra_df = st.checkbox("Compare extra DFs", value=True)

    # All base plot implementations expect 'gdf', 'pdf', or 'both'
    plot_kind = st.selectbox("Plot kind", ["gdf", "pdf", "both"], index=2)

    # Actions
    run_cols = st.columns(2)
    with run_cols[0]:
        do_fit = st.button("Fit Model", type="primary")
    with run_cols[1]:
        do_plot = st.button("Plot Results")

    # Persist fitted objects across reruns
    if "gdf_state" not in st.session_state:
        st.session_state["gdf_state"] = {}
    # Persist logs across reruns
    if "gdf_logs" not in st.session_state:
        st.session_state["gdf_logs"] = ""

    # Instantiate selected class
    gdf_obj = None
    if gdf_choice == "ELDF":
        gdf_obj = ELDF(
            DLB=DLB, DUB=DUB, LB=LB, UB=UB,
            S=S,
            varS=bool(varS) if varS is not None else False,
            minimum_varS=minimum_varS if minimum_varS is not None else 0.1,
            z0_optimize=z0_optimize,
            tolerance=tolerance if tolerance is not None else 1e-9,
            data_form=data_form,
            n_points=n_points,
            homogeneous=homogeneous,
            catch=catch,
            weights=weights,
            wedf=wedf,
            opt_method=opt_method,
            verbose=verbose,
            max_data_size=max_data_size,
            flush=flush,
        )
    elif gdf_choice == "EGDF":
        gdf_obj = EGDF(
            DLB=DLB, DUB=DUB, LB=LB, UB=UB,
            S=S,
            z0_optimize=z0_optimize,
            tolerance=tolerance if tolerance is not None else 1e-9,
            data_form=data_form,
            n_points=n_points,
            homogeneous=homogeneous,
            catch=catch,
            weights=weights,
            wedf=wedf,
            opt_method=opt_method,
            verbose=verbose,
            max_data_size=max_data_size,
            flush=flush,
        )
    elif gdf_choice == "QLDF":
        gdf_obj = QLDF(
            DLB=DLB, DUB=DUB, LB=LB, UB=UB,
            S=S,
            varS=bool(varS) if varS is not None else False,
            minimum_varS=minimum_varS if minimum_varS is not None else 0.1,
            z0_optimize=z0_optimize,
            tolerance=tolerance if tolerance is not None else 1e-9,
            data_form=data_form,
            n_points=n_points,
            homogeneous=homogeneous,
            catch=catch,
            weights=weights,
            wedf=wedf,
            opt_method=opt_method,
            verbose=verbose,
            max_data_size=max_data_size,
            flush=flush,
        )
    else:  # QGDF
        gdf_obj = QGDF(
            DLB=DLB, DUB=DUB, LB=LB, UB=UB,
            S=S,
            z0_optimize=z0_optimize,
            tolerance=tolerance if tolerance is not None else 1e-9,
            data_form=data_form,
            n_points=n_points,
            homogeneous=homogeneous,
            catch=catch,
            weights=weights,
            wedf=wedf,
            opt_method=opt_method,
            verbose=verbose,
            max_data_size=max_data_size,
            flush=flush,
        )

    # Auto-clear cached object if data length changed for this class
    if "gdf_meta" not in st.session_state:
        st.session_state["gdf_meta"] = {}
    prev_meta = st.session_state["gdf_meta"].get(gdf_choice, {})
    prev_obj = st.session_state["gdf_state"].get(gdf_choice)
    if prev_obj is not None and prev_meta.get("data_size") is not None and prev_meta["data_size"] != int(data.size):
        # Data length changed; clear cached fitted object to avoid weight-size mismatches
        st.session_state["gdf_state"].pop(gdf_choice, None)
        prev_obj = None
    # Update meta with current data size
    st.session_state["gdf_meta"][gdf_choice] = {"data_size": int(data.size)}

    # If we have a previously fitted object for this class, reuse it
    if prev_obj is not None:
        gdf_obj = prev_obj

    # Fit and show results
    if do_fit:
        if data.size == 0:
            st.error("No data parsed. Please enter valid numeric values.")
        else:
            try:
                # Capture stdout, stderr, and logging while fitting
                buf = io.StringIO()
                root_logger = logging.getLogger()
                mg_logger = logging.getLogger("machinegnostics")
                prev_root_level = root_logger.level
                prev_mg_level = mg_logger.level
                handler = logging.StreamHandler(buf)
                handler.setLevel(logging.DEBUG if verbose else logging.INFO)
                # Ensure logger levels allow messages through
                root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
                mg_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
                root_logger.addHandler(handler)
                mg_logger.addHandler(handler)
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    gdf_obj.fit(data=data, plot=False)
                st.success("Fitting completed.")
                # Cache fitted object for reuse on Plot click
                st.session_state["gdf_state"][gdf_choice] = gdf_obj
                # Save logs
                try:
                    root_logger.removeHandler(handler)
                    mg_logger.removeHandler(handler)
                    # Restore previous logger levels
                    root_logger.setLevel(prev_root_level)
                    mg_logger.setLevel(prev_mg_level)
                except Exception:
                    pass
                st.session_state["gdf_logs"] = buf.getvalue()
                try:
                    results = gdf_obj.results()
                    st.subheader("Results")
                    st.json(results)
                except Exception:
                    st.caption("Results method not available or failed; showing fitted object state.")
                    st.write(vars(gdf_obj))
            except Exception as e:
                st.error(f"Fit failed: {e}")

    # Reset button handling

    # Plot
    if do_plot:
        # Use cached fitted object if available
        gdf_obj = st.session_state["gdf_state"].get(gdf_choice, gdf_obj)
        # Guard: must be fitted before plotting
        if not hasattr(gdf_obj, "_fitted") or not getattr(gdf_obj, "_fitted"):
            st.error("Model not fitted yet. Please click 'Fit Model' first.")
        else:
            try:
                # Capture logs while plotting
                buf = io.StringIO()
                root_logger = logging.getLogger()
                mg_logger = logging.getLogger("machinegnostics")
                prev_root_level = root_logger.level
                prev_mg_level = mg_logger.level
                handler = logging.StreamHandler(buf)
                handler.setLevel(logging.DEBUG if verbose else logging.INFO)
                root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
                mg_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
                root_logger.addHandler(handler)
                mg_logger.addHandler(handler)
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    plt.close("all")
                    gdf_obj.plot(plot_smooth=plot_smooth, plot=plot_kind, bounds=bounds, extra_df=extra_df, figsize=(10, 6))
                fig = plt.gcf()
                st.pyplot(fig)
                # Save logs
                try:
                    root_logger.removeHandler(handler)
                    mg_logger.removeHandler(handler)
                    root_logger.setLevel(prev_root_level)
                    mg_logger.setLevel(prev_mg_level)
                except Exception:
                    pass
                st.session_state["gdf_logs"] = buf.getvalue()
            except Exception as e:
                st.error(f"Plot failed: {e}")

    # Logs panel
    with st.expander("Training Logs", expanded=True):
        logs = st.session_state.get("gdf_logs", "")
        if logs.strip():
            st.text(logs)
        else:
            st.caption("No logs captured yet. Enable 'verbose' for detailed output and fit/plot to populate logs.")


if __name__ == "__main__":
    main()
