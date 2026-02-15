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


def _parse_matrix(text: str) -> np.ndarray:
    """Parse multi-line matrix input: each line is a sample, values comma/space-separated."""
    if not text:
        return np.array([])
    rows = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if line == "":
            continue
        # Remove enclosing brackets if present
        line = line.replace("[", "").replace("]", "")
        # Split by comma or whitespace
        parts = [p for p in line.replace("\t", " ").replace(",", " ").split() if p]
        try:
            row = [float(p) for p in parts]
            rows.append(row)
        except ValueError:
            # skip malformed lines
            continue
    if not rows:
        return np.array([])
    # Ensure consistent number of columns
    ncols = len(rows[0])
    rows = [r for r in rows if len(r) == ncols]
    return np.asarray(rows, dtype=float)


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
    default_X = (
        "-0.14877078, 0.78315139\n"
        "0.51076183, 0.17344784\n"
        "0.64437712, -0.42849565\n"
        "-0.53732321, 0.59258391\n"
        "0.22222702, 0.29191072\n"
        "0.23572707, 1.20037615\n"
        "0.62012957, 0.6112227\n"
        "0.80739956, 0.56024128\n"
        "-0.07737646, 0.11472452\n"
        "0.60976333, 1.05652725\n"
        "1.06094342, -0.20799682\n"
        "1.02333716, -0.45626228\n"
        "-0.25819368, 0.89098369\n"
        "1.56966645, -0.38745305\n"
        "0.42832952, 0.418717\n"
        "1.12501815, 0.41063388\n"
        "1.39680669, -0.4302088\n"
        "1.68131368, -0.15557384\n"
        "0.7147671, -0.29233801\n"
        "-0.98663443, 0.36345703\n"
        "-0.91745348, 0.0861642\n"
        "1.80689434, -0.06165931\n"
        "-0.66046227, 0.64564557\n"
        "1.82683378, 0.69365567\n"
        "-0.86846608, 0.29560975\n"
        "1.49040032, -0.18742405\n"
        "0.09350187, 0.82814151\n"
        "0.19537903, -0.2915211\n"
        "0.21161396, -0.15171292\n"
        "1.91989946, 0.57646733"
    )
    default_y = (
        "0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1"
    )

    if "logr_X_text" not in st.session_state:
        st.session_state["logr_X_text"] = default_X
    if "logr_y_text" not in st.session_state:
        st.session_state["logr_y_text"] = default_y

    c_data = st.columns(2)
    with c_data[0]:
        X_text = st.text_area(
            "X values (lines; comma-separated features)",
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

    # Parse X as matrix (supports 1D if single value per line)
    X = _parse_matrix(X_text)
    # Accept 0/1 labels directly; if floats are provided, binarize with threshold 0.5
    y_raw = _parse_numbers(y_text)
    if y_raw.size > 0 and np.all(np.isin(y_raw, [0.0, 1.0])):
        y = y_raw.astype(int)
    else:
        y = _parse_labels(y_text)
    shape_txt = f"{X.shape[0]}x{X.shape[1]}" if X.ndim == 2 and X.size > 0 else f"{X.size}"
    st.caption(f"Parsed X: {shape_txt} · y: {y.size} labels")

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
    a_cols = st.columns(4)
    with a_cols[0]:
        do_fit = st.button("Fit Model", type="primary")
    with a_cols[1]:
        do_plot_prob = st.button("Plot Probabilities")
    with a_cols[2]:
        do_plot_w = st.button("Plot Weights")
    with a_cols[3]:
        do_plot_hist = st.button("Plot Loss & Entropy")

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
        elif (X.ndim == 1 and X.size != y.size) or (X.ndim == 2 and X.shape[0] != y.size):
            st.error("X and y must have the same number of points.")
        elif not np.all(np.isin(y, [0, 1])):
            st.error("Labels must be binary (0 or 1).")
        else:
            try:
                X_ = X.reshape(-1, 1) if X.ndim == 1 else X
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
                if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
                    X_plot = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
                else:
                    # Vary first feature, fix others at their means
                    mins = np.min(X, axis=0)
                    maxs = np.max(X, axis=0)
                    means = np.mean(X, axis=0)
                    x1 = np.linspace(mins[0], maxs[0], 200)
                    X_plot = np.column_stack([x1] + [np.full_like(x1, m) for m in means[1:]])
                y_proba = model.predict_proba(X_plot)
                fig, ax = plt.subplots(figsize=(10, 6))
                # Scatter of original labels on first feature
                x_plot_scatter = X if X.ndim == 1 else X[:, 0]
                ax.scatter(x_plot_scatter, y, c=y, cmap="coolwarm", s=100, label="Labels (0/1)", alpha=0.7)
                ax.plot(X_plot[:, 0], y_proba, "g-", linewidth=2, label="Predicted Probability")
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

    # Footer
    st.markdown("---")
    st.markdown("**Author**: Nirmal Parmar, [Machine Gnostics](https://machinegnostics.info)")

    # Plot training history: gnostic loss and residual entropy
    if do_plot_hist:
        model = st.session_state["logr_state"].get("LOGR", model)
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
