import io
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from contextlib import redirect_stdout, redirect_stderr

from machinegnostics.models import PolynomialRegressor


def _parse_matrix(text: str) -> np.ndarray:
    if not text:
        return np.empty((0, 0), dtype=float)
    rows = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"\s+", ",", line)
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 0), dtype=float)
    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        return np.empty((0, 0), dtype=float)
    return np.asarray(rows, dtype=float)


def _parse_numbers(text: str) -> np.ndarray:
    if not text:
        return np.array([], dtype=float)
    norm = re.sub(r"[\n\t\s]+", ",", text)
    vals = []
    for p in [s.strip() for s in norm.split(",") if s.strip() != ""]:
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return np.asarray(vals, dtype=float)


def _parse_int(text: str, default: int = None):
    try:
        return int(text)
    except Exception:
        return default


def _parse_float(text: str, default: float = None):
    try:
        return float(text)
    except Exception:
        return default


def main():
    st.set_page_config(page_title="Polynomial Regression | Machine Gnostics", layout="wide")

    st.title("Gnostic Polynomial Regression")
    st.markdown("Fit a robust polynomial model with multivariate features, visualize partial predictions, inspect automatic weights, and analyze training history.")

    # Learn panel
    with st.container(border=True):
        st.subheader("Learn")
        tabs = st.tabs(["Overview", "How It Works", "Data Entry", "Tips"])
        with tabs[0]:
            st.markdown(
                """
                The Machine Gnostics `PolynomialRegressor` extends robust regression to higher-order relationships.
                It optimizes gnostic criteria and automatically down-weights unreliable samples.
                """
            )
        with tabs[1]:
            st.markdown(
                """
                - Choose polynomial `degree` judiciously to avoid overfitting.
                - Dynamic weights reflect sample fidelity/irrelevance under gnostic geometry.
                - Training `history` records loss (`h_loss`) and residual entropy (`rentropy`).
                """
            )
        with tabs[2]:
            st.markdown(
                """
                Paste `X` as CSV rows (multiple columns) and `y` as a 1D list.
                Example `X` (2 features):\n\n1.0, 2.5\n0.7, -1.2\n1.3, 0.4\n\nExample `y`:\n\n3.1, -0.2, 1.8\n\nEnsure row count of `X` equals length of `y`.
                """
            )
        with tabs[3]:
            st.markdown(
                """
                - Start with small `degree` (e.g., 2 or 3); increase gradually.
                - Enable `verbose` to inspect solver logs when tuning parameters.
                - Use prediction plot to vary one feature (others fixed to mean) and observe curvature.
                - Inspect weights to verify robust behavior on outliers.
                """
            )

    # Data inputs
    st.subheader("Data")
    if "pr_X_text" not in st.session_state:
        # Default to 1D X for simplicity; multivariate still supported via CSV rows
        st.session_state["pr_X_text"] = "0.0\n0.4\n0.9\n1.4\n1.8"
    if "pr_y_text" not in st.session_state:
        st.session_state["pr_y_text"] = "2.1, 3.0, 4.6, 6.9, 9.8"

    c_data = st.columns(2)
    with c_data[0]:
        X_text = st.text_area(
            "X values (CSV rows; multiple columns)",
            height=160,
            placeholder="e.g., 1.0, 2.5\n0.7, -1.2\n1.3, 0.4",
            key="pr_X_text",
        )
    with c_data[1]:
        y_text = st.text_area(
            "y values (1D)",
            height=160,
            placeholder="e.g., 3.1, -0.2, 1.8",
            key="pr_y_text",
        )

    X = _parse_matrix(X_text)
    y = _parse_numbers(y_text)
    st.caption(f"Parsed X shape: {X.shape} · y length: {y.size}")
    if X.size > 0:
        st.dataframe(X[: min(5, X.shape[0]), :], use_container_width=True, height=160, hide_index=True)

    # Options
    st.subheader("Options")
    c_opts1 = st.columns(4)
    with c_opts1[0]:
        degree = _parse_int(st.text_input("degree", value="2"), default=2)
    with c_opts1[1]:
        scale_in = st.text_input("scale ('auto' or numeric)", value="auto")
        scale = scale_in if scale_in.strip() == "auto" else _parse_float(scale_in, default="auto")
    with c_opts1[2]:
        max_iter = _parse_int(st.text_input("max_iter", value="200"), default=200)
    with c_opts1[3]:
        tolerance = _parse_float(st.text_input("tolerance", value="1e-3"), default=1e-3)

    c_opts2 = st.columns(4)
    with c_opts2[0]:
        mg_loss = st.selectbox("mg_loss", ["hi", "hj"], index=0)
    with c_opts2[1]:
        early_stopping = st.checkbox("early_stopping", value=True)
    with c_opts2[2]:
        verbose = st.checkbox("verbose", value=False)
    with c_opts2[3]:
        data_form = st.selectbox("data_form", ["a", "m"], index=0)

    c_opts3 = st.columns(4)
    with c_opts3[0]:
        gnostic_characteristics = st.checkbox("gnostic_characteristics", value=False)

    # Actions
    a_cols = st.columns(4)
    with a_cols[0]:
        do_fit = st.button("Fit Model", type="primary")
    with a_cols[1]:
        do_plot_pred = st.button("Plot Predictions")
    with a_cols[2]:
        do_plot_w = st.button("Plot Weights")
    with a_cols[3]:
        do_plot_hist = st.button("Plot Loss & Entropy")

    # Session cache
    if "pr_state" not in st.session_state:
        st.session_state["pr_state"] = {}
    if "pr_meta" not in st.session_state:
        st.session_state["pr_meta"] = {}
    if "pr_logs" not in st.session_state:
        st.session_state["pr_logs"] = ""

    # Instantiate
    model = PolynomialRegressor(
        degree=degree,
        scale=scale,
        max_iter=max_iter,
        tolerance=tolerance,
        mg_loss=mg_loss,
        early_stopping=early_stopping,
        verbose=verbose,
        data_form=data_form,
        gnostic_characteristics=gnostic_characteristics,
        history=True,
    )

    # Auto-clear cached model when sample count changes
    prev_meta = st.session_state["pr_meta"].get("PR", {})
    prev_obj = st.session_state["pr_state"].get("PR")
    if prev_obj is not None and prev_meta.get("n_samples") is not None and prev_meta["n_samples"] != int(X.shape[0]):
        st.session_state["pr_state"].pop("PR", None)
        prev_obj = None
    st.session_state["pr_meta"]["PR"] = {"n_samples": int(X.shape[0])}
    if prev_obj is not None:
        model = prev_obj

    # Fit
    if do_fit:
        if X.size == 0:
            st.error("X is empty or could not be parsed. Ensure consistent CSV columns.")
        elif y.size == 0:
            st.error("y is empty. Provide 1D numeric values.")
        elif X.shape[0] != y.size:
            st.error(f"Mismatch: X rows = {X.shape[0]} vs y length = {y.size}.")
        else:
            try:
                log_buf = io.StringIO()
                root_logger = logging.getLogger()
                mg_logger = logging.getLogger("machinegnostics")
                handler = logging.StreamHandler(log_buf)
                handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
                if verbose:
                    root_prev = root_logger.level
                    mg_prev = mg_logger.level
                    root_logger.setLevel(logging.DEBUG)
                    mg_logger.setLevel(logging.DEBUG)
                    root_logger.addHandler(handler)
                    mg_logger.addHandler(handler)
                    try:
                        with redirect_stdout(log_buf), redirect_stderr(log_buf):
                            model.fit(X, y)
                    finally:
                        root_logger.removeHandler(handler)
                        mg_logger.removeHandler(handler)
                        root_logger.setLevel(root_prev)
                        mg_logger.setLevel(mg_prev)
                else:
                    with st.spinner("Fitting..."):
                        model.fit(X, y)
                st.session_state["pr_logs"] = log_buf.getvalue()
                st.success("Model fitted successfully.")
                st.session_state["pr_state"]["PR"] = model
                st.subheader("Results")
                st.write({
                    "degree": degree,
                    "n_features": X.shape[1],
                    "coefficients": getattr(model, "coefficients", None),
                    "tolerance": model.tolerance,
                    "max_iter": model.max_iter,
                    "mg_loss": model.mg_loss,
                })
            except Exception as e:
                st.error(f"Fit failed: {e}")

    with st.expander("Training Logs", expanded=False):
        logs = st.session_state.get("pr_logs", "")
        if logs:
            st.text(logs)
        else:
            st.write("No logs captured yet. Enable verbose and fit the model to see detailed logs.")

    # Plot predictions (vary one feature)
    if do_plot_pred:
        model = st.session_state["pr_state"].get("PR", model)
        try:
            if X.size == 0 or y.size == 0:
                st.error("Provide both X and y before plotting.")
            elif not hasattr(model, "predict"):
                st.error("Model not fitted. Fit the model first.")
            else:
                plt.close("all")
                n_features = X.shape[1] if X.ndim == 2 else 1
                feature_idx = st.number_input("Feature index to vary (0-based)", min_value=0, max_value=max(0, n_features - 1), value=0)
                base = np.mean(X, axis=0)
                f_min, f_max = float(np.min(X[:, int(feature_idx)])), float(np.max(X[:, int(feature_idx)]))
                grid = np.linspace(f_min, f_max, 200)
                X_plot = np.tile(base, (grid.size, 1))
                X_plot[:, int(feature_idx)] = grid
                y_pred = model.predict(X_plot)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X[:, int(feature_idx)], y, color="blue", s=90, label="Data", alpha=0.6)
                ax.plot(grid, y_pred, "g-", linewidth=2, label="Predicted (varying feature)")
                ax.set_xlabel(f"Feature {int(feature_idx)}")
                ax.set_ylabel("y")
                ax.set_title(f"Polynomial Regression (degree={degree}) - Partial Dependence")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Plot predictions failed: {e}")

    # Plot weights
    if do_plot_w:
        model = st.session_state["pr_state"].get("PR", model)
        try:
            weights = getattr(model, "weights", None)
            if weights is None:
                st.error("Model not fitted or weights unavailable. Fit the model first.")
            else:
                plt.close("all")
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(range(len(weights)), weights, color="steelblue", edgecolor="black", alpha=0.75)
                if len(weights) > 0:
                    idx_min = int(np.argmin(weights))
                    bars[idx_min].set_color("red")
                    bars[idx_min].set_alpha(0.85)
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Gnostic Weight")
                ax.set_title("Automatic Sample Weights")
                ax.grid(True, axis="y", alpha=0.3)
                ax.set_xticks(range(len(weights)))
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Plot weights failed: {e}")

    # Plot training history: loss & entropy
    if do_plot_hist:
        model = st.session_state["pr_state"].get("PR", model)
        try:
            hist = getattr(model, "_history", None)
            if not hist:
                st.error("No training history available. Fit the model with history=True.")
            else:
                iterations = [int(h.get("iteration", i + 1)) for i, h in enumerate(hist)]
                h_loss = [h.get("h_loss", np.nan) for h in hist]
                rentropy = [h.get("rentropy", np.nan) for h in hist]

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].plot(iterations, h_loss, marker="o", color="tab:blue", linewidth=2)
                axes[0].set_title("Gnostic Loss (h_loss)")
                axes[0].set_xlabel("Iteration")
                axes[0].set_ylabel("Loss")
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(iterations, rentropy, marker="s", color="tab:orange", linewidth=2)
                axes[1].set_title("Residual Entropy (rentropy)")
                axes[1].set_xlabel("Iteration")
                axes[1].set_ylabel("Entropy")
                axes[1].grid(True, alpha=0.3)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Plot history failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown("**Author**: Nirmal Parmar, [Machine Gnostics](https://machinegnostics.info)")


if __name__ == "__main__":
    main()
