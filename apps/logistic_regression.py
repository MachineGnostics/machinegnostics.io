import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from machinegnostics.models import LogisticRegressor


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


def _parse_labels(text: str) -> np.ndarray:
    arr = _parse_numbers(text)
    if arr.size == 0:
        return arr
    # Convert to binary labels {0,1}
    labels = np.array([1 if x >= 0.5 else 0 for x in arr], dtype=int)
    return labels


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
    st.set_page_config(page_title="Logistic Regression | Machine Gnostics", layout="wide")

    st.title("Gnostic Logistic Regression")
    st.markdown("Fit a robust logistic model, visualize probabilities, and inspect automatic weights.")

    # Learn panel
    with st.container(border=True):
        st.subheader("Learn: Robust Logistic Regression")
        tabs = st.tabs(["Overview", "Tips"])
        with tabs[0]:
            st.markdown(
                """
                The Machine Gnostics `LogisticRegressor` performs binary classification with robust optimization,
                optional polynomial feature expansion, and automatic down-weighting of outliers.
                """
            )
        with tabs[1]:
            st.markdown(
                """
                - Provide 1D inputs for `X` and matching binary labels `y` (0/1).
                - Inspect the probability curve and weights to understand model behavior.
                - Start with default parameters; increase `degree` or iterations gradually.
                """
            )

    # Data inputs
    st.subheader("Data")
    default_X = "0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0"
    # Labels encoded roughly by threshold; include one flipped label to simulate noise
    default_y = "0, 0, 1, 1, 1, 1, 1, 0, 1"

    if "logr_X_text" not in st.session_state:
        st.session_state["logr_X_text"] = default_X
    if "logr_y_text" not in st.session_state:
        st.session_state["logr_y_text"] = default_y

    c_data = st.columns(2)
    with c_data[0]:
        X_text = st.text_area(
            "X values (1D)",
            height=100,
            placeholder="e.g., 0, 0.5, 1.0, 1.5, 2.0",
            key="logr_X_text",
        )
    with c_data[1]:
        y_text = st.text_area(
            "y labels (0/1)",
            height=100,
            placeholder="e.g., 0, 0, 1, 1, 1",
            key="logr_y_text",
        )

    X = _parse_numbers(X_text)
    # Accept 0/1 labels directly; if floats are provided, binarize with threshold 0.5
    y_raw = _parse_numbers(y_text)
    if y_raw.size > 0 and np.all(np.isin(y_raw, [0.0, 1.0])):
        y = y_raw.astype(int)
    else:
        y = _parse_labels(y_text)
    st.caption(f"Parsed X: {X.size} points · y: {y.size} labels")

    # Options
    st.subheader("Options")
    c_opts1 = st.columns(4)
    with c_opts1[0]:
        degree = _parse_int(st.text_input("degree", value="1"), default=1)
    with c_opts1[1]:
        max_iter = _parse_int(st.text_input("max_iter", value="100"), default=100)
    with c_opts1[2]:
        tolerance = _parse_float(st.text_input("tolerance", value="1e-2"), default=1e-2)
    with c_opts1[3]:
        early_stopping = st.checkbox("early_stopping", value=True)

    c_opts2 = st.columns(4)
    with c_opts2[0]:
        verbose = st.checkbox("verbose", value=False)
    with c_opts2[1]:
        scale_in = st.text_input("scale ('auto' or numeric)", value="auto")
        scale = scale_in if scale_in.strip() == "auto" else _parse_float(scale_in, default="auto")
    with c_opts2[2]:
        data_form = st.selectbox("data_form", ["a", "m"], index=0)
    with c_opts2[3]:
        gnostic_characteristics = st.checkbox("gnostic_characteristics", value=False)

    # Actions
    a_cols = st.columns(3)
    with a_cols[0]:
        do_fit = st.button("Fit Model", type="primary")
    with a_cols[1]:
        do_plot_prob = st.button("Plot Probabilities")
    with a_cols[2]:
        do_plot_w = st.button("Plot Weights")

    # Session cache
    if "logr_state" not in st.session_state:
        st.session_state["logr_state"] = {}
    if "logr_meta" not in st.session_state:
        st.session_state["logr_meta"] = {}

    # Instantiate
    model = LogisticRegressor(
        degree=degree,
        max_iter=max_iter,
        tolerance=tolerance,
        early_stopping=early_stopping,
        verbose=verbose,
        scale=scale,
        data_form=data_form,
        gnostic_characteristics=gnostic_characteristics,
        history=True,
    )

    # Auto-clear cached model when data size changes
    prev_meta = st.session_state["logr_meta"].get("LOGR", {})
    prev_obj = st.session_state["logr_state"].get("LOGR")
    if prev_obj is not None and prev_meta.get("data_size") is not None and prev_meta["data_size"] != int(X.size):
        st.session_state["logr_state"].pop("LOGR", None)
        prev_obj = None
    st.session_state["logr_meta"]["LOGR"] = {"data_size": int(X.size)}
    if prev_obj is not None:
        model = prev_obj

    # Fit
    if do_fit:
        if X.size == 0 or y.size == 0:
            st.error("Please provide both X and y labels.")
        elif X.size != y.size:
            st.error("X and y must have the same number of points.")
        elif not np.all(np.isin(y, [0, 1])):
            st.error("Labels must be binary (0 or 1).")
        else:
            try:
                X_ = X.reshape(-1, 1)
                model.fit(X_, y)
                st.success("Model fitted successfully.")
                st.session_state["logr_state"]["LOGR"] = model
                # Show quick summary
                try:
                    st.subheader("Results")
                    st.write({
                        "degree": degree,
                        "tolerance": model.tolerance,
                        "max_iter": model.max_iter,
                        "coefficients": getattr(model, "coefficients", None),
                    })
                except Exception:
                    st.write(vars(model))
            except Exception as e:
                st.error(f"Fit failed: {e}")

    # Plot probabilities
    if do_plot_prob:
        model = st.session_state["logr_state"].get("LOGR", model)
        try:
            if X.size == 0 or y.size == 0:
                st.error("Please provide both X and y labels.")
            else:
                plt.close("all")
                X_plot = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
                y_proba = model.predict_proba(X_plot)
                fig, ax = plt.subplots(figsize=(10, 6))
                # Scatter of original labels
                ax.scatter(X, y, c=y, cmap="coolwarm", s=100, label="Labels (0/1)", alpha=0.7)
                ax.plot(X_plot.ravel(), y_proba, "g-", linewidth=2, label="Predicted Probability")
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision Threshold 0.5")
                ax.set_xlabel("X")
                ax.set_ylabel("Probability")
                ax.set_title("Logistic Regression Probabilities")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Plot probabilities failed: {e}")

    # Plot weights
    if do_plot_w:
        model = st.session_state["logr_state"].get("LOGR", model)
        try:
            if not hasattr(model, "weights") or model.weights is None:
                st.error("Model not fitted or weights unavailable. Fit the model first.")
            else:
                plt.close("all")
                weights = model.weights
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(range(len(weights)), weights, color="steelblue", edgecolor="black", alpha=0.7)
                # Highlight lowest weight
                if len(weights) > 0:
                    idx_min = int(np.argmin(weights))
                    bars[idx_min].set_color("red")
                    bars[idx_min].set_alpha(0.85)
                ax.set_xlabel("Data Point Index")
                ax.set_ylabel("Gnostic Weight")
                ax.set_title("Automatic Sample Weights")
                ax.grid(True, axis="y", alpha=0.3)
                ax.set_xticks(range(len(weights)))
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Plot weights failed: {e}")

    st.markdown("---")
    st.markdown("**Author**: Nirmal Parmar, [Machine Gnostics](https://machinegnostics.info)")
    
if __name__ == "__main__":
    main()
