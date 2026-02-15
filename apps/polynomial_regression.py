import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from machinegnostics.models import PolynomialRegressor


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
    st.markdown("Fit a robust polynomial model, visualize predictions, and inspect automatic weights.")

    # Learn panel
    with st.container(border=True):
        st.subheader("Learn: Robust Polynomial Regression")
        tabs = st.tabs(["Overview", "Tips"])
        with tabs[0]:
            st.markdown(
                """
                The Machine Gnostics `PolynomialRegressor` extends robust regression to higher-order polynomials,
                optimizing gnostic criteria and automatically down-weighting outliers.
                """
            )
        with tabs[1]:
            st.markdown(
                """
                - Provide 1D inputs for `X` and matching `y` values.
                - Choose polynomial `degree` and keep defaults for stability; increase gradually.
                - Inspect weights to see automatic down-weighting of outliers.
                """
            )

    # Data inputs
    st.subheader("Data")
    default_X = "0.0, 0.22222222, 0.44444444, 0.66666667, 0.88888889, 1.11111111, 1.33333333, 1.55555556, 1.77777778, 2.0"
    default_y = "2.09934283, 2.95599654, 4.58061956, 6.94483982, 9.85923417, 14.73128481, 22.36219532, 33.04278049, 128.97116552, 65.3049809"

    if "pr_X_text" not in st.session_state:
        st.session_state["pr_X_text"] = default_X
    if "pr_y_text" not in st.session_state:
        st.session_state["pr_y_text"] = default_y

    c_data = st.columns(2)
    with c_data[0]:
        X_text = st.text_area(
            "X values (1D)",
            height=100,
            placeholder="e.g., 0, 0.5, 1.0, 1.5, 2.0",
            key="pr_X_text",
        )
    with c_data[1]:
        y_text = st.text_area(
            "y values",
            height=100,
            placeholder="e.g., 1, 6, 11, 16, 21",
            key="pr_y_text",
        )

    X = _parse_numbers(X_text)
    y = _parse_numbers(y_text)
    st.caption(f"Parsed X: {X.size} points · y: {y.size} points")

    # Options
    st.subheader("Options")
    c_opts1 = st.columns(4)
    with c_opts1[0]:
        degree = _parse_int(st.text_input("degree", value="2"), default=2)
    with c_opts1[1]:
        scale_in = st.text_input("scale ('auto' or numeric)", value="auto")
        scale = scale_in if scale_in.strip() == "auto" else _parse_float(scale_in, default="auto")
    with c_opts1[2]:
        max_iter = _parse_int(st.text_input("max_iter", value="100"), default=100)
    with c_opts1[3]:
        tolerance = _parse_float(st.text_input("tolerance", value="1e-2"), default=1e-2)

    c_opts2 = st.columns(4)
    with c_opts2[0]:
        mg_loss = st.selectbox("mg_loss", ["hi", "fi"], index=0)
    with c_opts2[1]:
        early_stopping = st.checkbox("early_stopping", value=True)
    with c_opts2[2]:
        verbose = st.checkbox("verbose", value=False)
    with c_opts2[3]:
        data_form = st.selectbox("data_form", ["a", "m"], index=0)

    c_opts3 = st.columns(4)
    with c_opts3[0]:
        gnostic_characteristics = st.checkbox("gnostic_characteristics", value=False)
    # remaining slots unused for symmetry

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

    # Auto-clear cached model when data size changes
    prev_meta = st.session_state["pr_meta"].get("PR", {})
    prev_obj = st.session_state["pr_state"].get("PR")
    if prev_obj is not None and prev_meta.get("data_size") is not None and prev_meta["data_size"] != int(X.size):
        st.session_state["pr_state"].pop("PR", None)
        prev_obj = None
    st.session_state["pr_meta"]["PR"] = {"data_size": int(X.size)}
    if prev_obj is not None:
        model = prev_obj

    # Fit
    if do_fit:
        if X.size == 0 or y.size == 0:
            st.error("Please provide both X and y values.")
        elif X.size != y.size:
            st.error("X and y must have the same number of points.")
        else:
            try:
                X_ = X.reshape(-1, 1)
                model.fit(X_, y)
                st.success("Model fitted successfully.")
                st.session_state["pr_state"]["PR"] = model
                # Show coefficients summary
                try:
                    st.subheader("Results")
                    st.write({
                        "degree": degree,
                        "coefficients": getattr(model, "coefficients", None),
                        "tolerance": model.tolerance,
                        "max_iter": model.max_iter,
                        "mg_loss": model.mg_loss,
                    })
                except Exception:
                    st.write(vars(model))
            except Exception as e:
                st.error(f"Fit failed: {e}")

    # Plot predictions
    if do_plot_pred:
        model = st.session_state["pr_state"].get("PR", model)
        try:
            if X.size == 0 or y.size == 0:
                st.error("Please provide both X and y values.")
            else:
                plt.close("all")
                X_plot = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
                y_pred = model.predict(X_plot)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X, y, color="blue", s=100, label="Data", alpha=0.6)
                ax.plot(X_plot.ravel(), y_pred, "g-", linewidth=2, label="Predicted")
                ax.set_xlabel("X")
                ax.set_ylabel("y")
                ax.set_title(f"Polynomial Regression Predictions (degree={degree})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Plot predictions failed: {e}")

    # Plot weights
    if do_plot_w:
        model = st.session_state["pr_state"].get("PR", model)
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

    # Footer
    st.markdown("---")
    st.markdown("**Author**: Nirmal Parmar, [Machine Gnostics](https://machinegnostics.info)")

    # Plot training history: gnostic loss and residual entropy
    if do_plot_hist:
        model = st.session_state["pr_state"].get("PR", model)
        try:
            hist = getattr(model, "_history", None)
            if not hist:
                st.error("No training history available. Fit the model with history=True.")
            else:
                history_valid = [h for h in hist if isinstance(h, dict) and ("h_loss" in h or "rentropy" in h)]
                if len(history_valid) == 0:
                    st.error("Training history does not contain loss/entropy fields.")
                else:
                    iterations = [h.get("iteration", i+1) for i, h in enumerate(history_valid)]
                    h_loss = [h.get("h_loss", np.nan) for h in history_valid]
                    rentropy = [h.get("rentropy", np.nan) for h in history_valid]

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].plot(iterations, h_loss, marker='o', color='tab:blue', linewidth=2)
                    axes[0].set_title('Gnostic Loss (h_loss)')
                    axes[0].set_xlabel('Iteration')
                    axes[0].set_ylabel('Loss')
                    axes[0].grid(True, alpha=0.3)

                    axes[1].plot(iterations, rentropy, marker='s', color='tab:orange', linewidth=2)
                    axes[1].set_title('Residual Entropy (rentropy)')
                    axes[1].set_xlabel('Iteration')
                    axes[1].set_ylabel('Entropy')
                    axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Plot history failed: {e}")


if __name__ == "__main__":
    main()
