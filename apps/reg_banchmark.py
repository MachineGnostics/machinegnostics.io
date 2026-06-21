"""Regression benchmark app for the Anscombe Quartet.

Compares classical regression models against Machine Gnostics regression models
on the four Anscombe datasets.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

try:
	from xgboost import XGBRegressor

	XGBOOST_AVAILABLE = True
except Exception:
	XGBRegressor = None  # type: ignore[assignment]
	XGBOOST_AVAILABLE = False

from machinegnostics.data import make_anscombe_check_data
from machinegnostics.metrics import robr2
from machinegnostics.models import (
	GnosticBoostingRegressor,
	GnosticDecisionTreeRegressor,
	GnosticRandomForestRegressor,
	LinearRegressor,
	PolynomialRegressor,
)


st.set_page_config(
	page_title="Regression Benchmark | Machine Gnostics",
	page_icon="📈",
	layout="wide",
	initial_sidebar_state="expanded",
)


st.markdown(
	"""
	<style>
	:root {
		--bg: #07111f;
		--panel: rgba(11, 21, 37, 0.94);
		--line: #23405f;
		--text: #e6edf7;
		--muted: #9db0c9;
		--cyan: #4fc3f7;
		--green: #67d17e;
		--amber: #ffb347;
		--red: #ff6b6b;
	}

	.stApp {
		background:
			radial-gradient(circle at top left, rgba(79, 195, 247, 0.12), transparent 26%),
			radial-gradient(circle at top right, rgba(103, 209, 126, 0.10), transparent 20%),
			linear-gradient(180deg, #07101d 0%, #08111f 55%, #091423 100%);
		color: var(--text);
	}

	[data-testid="stSidebar"] {
		background: linear-gradient(180deg, #07101d 0%, #0a1527 100%);
		border-right: 1px solid rgba(145, 180, 220, 0.15);
	}

	[data-testid="stSidebar"] * {
		color: var(--text) !important;
	}

	.hero {
		background: linear-gradient(135deg, rgba(19, 35, 61, 0.98), rgba(9, 20, 35, 0.96));
		border: 1px solid rgba(80, 199, 255, 0.20);
		border-radius: 20px;
		padding: 24px;
		box-shadow: 0 18px 50px rgba(0, 0, 0, 0.26);
		margin-bottom: 1rem;
	}

	.hero h1 {
		margin: 0 0 8px 0;
		color: var(--text);
		font-size: 2rem;
	}

	.hero p {
		color: var(--muted);
		margin: 0;
		line-height: 1.65;
	}

	.kpi {
		background: linear-gradient(180deg, rgba(19, 35, 61, 0.95), rgba(11, 22, 39, 0.95));
		border: 1px solid rgba(145, 180, 220, 0.16);
		border-radius: 16px;
		padding: 16px 14px;
		min-height: 118px;
	}

	.kpi-label {
		color: var(--muted);
		font-size: 0.76rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.kpi-value {
		color: var(--text);
		font-size: 1.65rem;
		font-weight: 700;
		margin-top: 8px;
		line-height: 1.1;
	}

	.kpi-sub {
		color: var(--muted);
		font-size: 0.84rem;
		margin-top: 4px;
		line-height: 1.35;
	}

	.good { color: var(--green) !important; }
	.warn { color: var(--amber) !important; }
	.bad { color: var(--red) !important; }

	.section-title {
		margin: 0 0 12px 0;
		font-size: 1.1rem;
		letter-spacing: 0.02em;
		color: var(--text);
	}

	.callout {
		background: linear-gradient(90deg, rgba(79, 195, 247, 0.12), rgba(103, 209, 126, 0.08));
		border: 1px solid rgba(80, 199, 255, 0.16);
		border-left: 4px solid var(--cyan);
		border-radius: 14px;
		padding: 14px 16px;
		color: var(--muted);
		line-height: 1.6;
	}

	.mini-note {
		color: var(--muted);
		font-size: 0.86rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


DS_NAMES = {
	1: "Anscombe I",
	2: "Anscombe II",
	3: "Anscombe III",
	4: "Anscombe IV",
}


def format_float(value: float | None, digits: int = 4) -> str:
	if value is None:
		return "-"
	try:
		if np.isnan(value):
			return "-"
	except Exception:
		pass
	return f"{value:.{digits}f}"


def as_2d(x: np.ndarray) -> np.ndarray:
	return np.asarray(x, dtype=float).reshape(-1, 1)


def pct_change(classical_val: float, model_val: float, lower_is_better: bool = False) -> tuple[float, str, str]:
	if abs(classical_val) < 1e-12:
		return 0.0, "0.0%", "kpi-sub"
	delta = ((model_val - classical_val) / abs(classical_val)) * 100.0
	if lower_is_better:
		delta = -delta
	label = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
	css = "good" if delta > 0 else ("warn" if delta > -5 else "bad")
	return delta, label, css


def kpi_card(label: str, model_val: float, classical_val: float, change_label: str, change_css: str, unit: str = "") -> str:
	return f"""
	<div class="kpi">
		<div class="kpi-label">{label}</div>
		<div class="kpi-value">{format_float(model_val)}{unit}</div>
		<div class="kpi-sub">Classical: {format_float(classical_val)}{unit}</div>
		<div class="kpi-sub {change_css}">Benchmark change: {change_label}</div>
	</div>
	"""


@st.cache_data(show_spinner=False)
def load_anscombe() -> dict[int, dict[str, np.ndarray]]:
	data: dict[int, dict[str, np.ndarray]] = {}
	for ds_id in [1, 2, 3, 4]:
		x, y = make_anscombe_check_data(ds_id)
		data[ds_id] = {"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}
	return data


def build_model_specs(
	poly_degree: int,
	rf_estimators: int,
	rf_depth: int,
	xgb_estimators: int,
	xgb_depth: int,
	random_state: int,
) -> list[dict[str, object]]:
	specs: list[dict[str, object]] = [
		{
			"name": "Linear OLS",
			"family": "classical",
			"color": "#4fc3f7",
			"builder": lambda: LinearRegression(),
		},
		{
			"name": f"Polynomial OLS (d={poly_degree})",
			"family": "classical",
			"color": "#ffb347",
			"builder": lambda: make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False), LinearRegression()),
		},
		{
			"name": "Random Forest",
			"family": "classical",
			"color": "#9b7bff",
			"builder": lambda: RandomForestRegressor(
				n_estimators=rf_estimators,
				max_depth=rf_depth if rf_depth > 0 else None,
				random_state=random_state,
			),
		},
	]

	if XGBOOST_AVAILABLE:
		specs.append(
			{
				"name": "XGBoost",
				"family": "classical",
				"color": "#ff6b6b",
				"builder": lambda: XGBRegressor(
					n_estimators=xgb_estimators,
					max_depth=xgb_depth,
					learning_rate=0.12,
					subsample=1.0,
					colsample_bytree=1.0,
					reg_lambda=1.0,
					random_state=random_state,
					objective="reg:squarederror",
					verbosity=0,
				),
			}
		)

	specs.extend(
		[
			{
				"name": "MG Linear",
				"family": "mg",
				"color": "#67d17e",
				"builder": lambda: LinearRegressor(
					max_iter=300,
					early_stopping=True,
					tolerance=1e-6,
					mg_loss="hi",
					history=True,
					verbose=False,
				),
			},
			{
				"name": f"MG Polynomial (d={poly_degree})",
				"family": "mg",
				"color": "#7fe0c3",
				"builder": lambda: PolynomialRegressor(
					degree=poly_degree,
					scale="auto",
					max_iter=300,
					tolerance=1e-6,
					mg_loss="hi",
					early_stopping=True,
					verbose=False,
					data_form="a",
					gnostic_characteristics=False,
					history=True,
				),
			},
			{
				"name": "MG Decision Tree Model",
				"family": "mg",
				"color": "#61d9a6",
				"builder": lambda: GnosticDecisionTreeRegressor(max_depth=3),
			},
			{
				"name": "MG Bagging Model",
				"family": "mg",
				"color": "#43cfa0",
				"builder": lambda: GnosticRandomForestRegressor(
					n_estimators=rf_estimators,
					max_depth=rf_depth if rf_depth > 0 else None,
					random_state=random_state,
				),
			},
			{
				"name": "MG Boosting Model",
				"family": "mg",
				"color": "#2bbf94",
				"builder": lambda: GnosticBoostingRegressor(
					n_estimators=xgb_estimators,
					max_depth=xgb_depth,
					random_state=random_state,
				),
			},
		]
	)
	return specs


def evaluate_model(model_spec: dict[str, object], x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
	model = model_spec["builder"]()  # type: ignore[index]
	X = as_2d(x)
	model.fit(X, y)
	y_pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
	weights = getattr(model, "weights", None)
	weights_arr = np.asarray(weights, dtype=float).reshape(-1) if weights is not None else None

	metrics = {
		"r2": float(r2_score(y, y_pred)),
		"rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
		"mae": float(mean_absolute_error(y, y_pred)),
		"corr": float(np.corrcoef(y, y_pred)[0, 1]) if len(y) > 1 else float("nan"),
		"robr2": float(robr2(y, y_pred, w=weights_arr)) if weights_arr is not None else float("nan"),
		"y_pred": y_pred,
		"weights": weights_arr,
		"model": model,
	}
	return metrics


def evaluate_dataset(x: np.ndarray, y: np.ndarray, model_specs: list[dict[str, object]]) -> pd.DataFrame:
	rows = []
	for spec in model_specs:
		result = evaluate_model(spec, x, y)
		rows.append({"Model": spec["name"], "Family": spec["family"], "Color": spec["color"], **result})
	frame = pd.DataFrame(rows)
	frame["R2 Rank"] = frame["r2"].rank(ascending=False, method="min")
	frame["RMSE Rank"] = frame["rmse"].rank(ascending=True, method="min")
	frame["Overall Rank"] = frame[["R2 Rank", "RMSE Rank"]].mean(axis=1)
	return frame.sort_values(["Overall Rank", "rmse", "r2"], ascending=[True, True, False]).reset_index(drop=True)


def dense_prediction_grid(x: np.ndarray) -> np.ndarray:
	x_min = float(np.min(x))
	x_max = float(np.max(x))
	pad = max(0.8, 0.18 * (x_max - x_min))
	return np.linspace(x_min - pad, x_max + pad, 250)


def plot_dataset_overview(x: np.ndarray, y: np.ndarray, title: str):
	fig, ax = plt.subplots(figsize=(6.8, 4.2))
	fig.patch.set_facecolor("#07111f")
	ax.set_facecolor("#07111f")
	ax.scatter(x, y, s=70, color="#4fc3f7", edgecolors="white", linewidths=0.5)
	ax.set_title(title, color="#e6edf7")
	ax.set_xlabel("X", color="#9db0c9")
	ax.set_ylabel("Y", color="#9db0c9")
	ax.tick_params(colors="#9db0c9")
	ax.grid(alpha=0.12)
	for spine in ax.spines.values():
		spine.set_edgecolor("#23405f")
	return fig


def plot_model_fits(x: np.ndarray, y: np.ndarray, model_frame: pd.DataFrame, title: str):
	grid_x = dense_prediction_grid(x)
	grid_X = as_2d(grid_x)

	fig, ax = plt.subplots(figsize=(10, 6))
	fig.patch.set_facecolor("#07111f")
	ax.set_facecolor("#07111f")
	ax.scatter(x, y, s=75, color="#d8f2ff", alpha=0.95, edgecolors="white", linewidths=0.5, label="Anscombe data", zorder=4)

	for _, row in model_frame.iterrows():
		y_grid = np.asarray(row["model"].predict(grid_X), dtype=float).reshape(-1)
		ax.plot(grid_x, y_grid, color=row["Color"], linewidth=2.3, label=row["Model"], zorder=3)

	ax.set_title(title, color="#e6edf7")
	ax.set_xlabel("X", color="#9db0c9")
	ax.set_ylabel("Y", color="#9db0c9")
	ax.tick_params(colors="#9db0c9")
	ax.grid(alpha=0.12)
	for spine in ax.spines.values():
		spine.set_edgecolor("#23405f")
	ax.legend(facecolor="#101c32", edgecolor="#244063", labelcolor="#e6edf7", fontsize=9, ncol=2)
	return fig


def plot_weights(x: np.ndarray, y: np.ndarray, model_frame: pd.DataFrame, title: str):
	mg_rows = model_frame[model_frame["Family"] == "mg"]
	fig, axes = plt.subplots(1, max(len(mg_rows), 1), figsize=(5.6 * max(len(mg_rows), 1), 4.4), squeeze=False)
	fig.patch.set_facecolor("#07111f")
	axes = axes[0]

	if mg_rows.empty:
		axes[0].text(0.5, 0.5, "No Machine Gnostics model selected", ha="center", va="center", color="#e6edf7")
		axes[0].set_axis_off()
		return fig

	for ax, (_, row) in zip(axes, mg_rows.iterrows()):
		weights = row["weights"]
		ax.set_facecolor("#07111f")
		if weights is None or len(weights) != len(x):
			sizes = np.full_like(x, 80, dtype=float)
			colors = np.full_like(x, 0.5, dtype=float)
		else:
			weights = np.asarray(weights, dtype=float)
			normalized = (weights - weights.min()) / (weights.max() - weights.min() + 1e-12)
			sizes = 70 + normalized * 220
			colors = weights

		sc = ax.scatter(x, y, s=sizes, c=colors, cmap="RdYlGn", alpha=0.92, edgecolors="white", linewidths=0.45)
		ax.set_title(row["Model"], color="#e6edf7")
		ax.set_xlabel("X", color="#9db0c9")
		ax.set_ylabel("Y", color="#9db0c9")
		ax.tick_params(colors="#9db0c9")
		ax.grid(alpha=0.12)
		for spine in ax.spines.values():
			spine.set_edgecolor("#23405f")
		cbar = fig.colorbar(sc, ax=ax)
		cbar.set_label("Adaptive weight", color="#9db0c9")
		cbar.ax.yaxis.set_tick_params(color="#9db0c9")
		plt.setp(cbar.ax.get_yticklabels(), color="#9db0c9")

	fig.suptitle(title, color="#e6edf7", y=1.03)
	fig.tight_layout()
	return fig


def pair_mapping(poly_degree: int) -> list[tuple[str, str]]:
	pairs = [
		("Linear OLS", "MG Linear"),
		(f"Polynomial OLS (d={poly_degree})", f"MG Polynomial (d={poly_degree})"),
		("Random Forest", "MG Bagging Model"),
		("XGBoost", "MG Boosting Model"),
	]
	return pairs


def evaluate_pairwise(
	x: np.ndarray,
	y: np.ndarray,
	model_specs: list[dict[str, object]],
	poly_degree: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
	spec_by_name = {str(spec["name"]): spec for spec in model_specs}
	rows: list[dict[str, object]] = []
	details: list[dict[str, object]] = []

	for classical_name, mg_name in pair_mapping(poly_degree):
		if classical_name not in spec_by_name or mg_name not in spec_by_name:
			continue

		classical_spec = spec_by_name[classical_name]
		mg_spec = spec_by_name[mg_name]

		classical_metrics = evaluate_model(classical_spec, x, y)
		mg_metrics = evaluate_model(mg_spec, x, y)

		r2_delta, r2_label, _ = pct_change(float(classical_metrics["r2"]), float(mg_metrics["r2"]))
		rmse_delta, rmse_label, _ = pct_change(
			float(classical_metrics["rmse"]),
			float(mg_metrics["rmse"]),
			lower_is_better=True,
		)

		rows.append(
			{
				"Pair": f"{classical_name} vs {mg_name}",
				"Classical Model": classical_name,
				"MG Model": mg_name,
				"Classical R2": float(classical_metrics["r2"]),
				"MG R2": float(mg_metrics["r2"]),
				"R2 Improvement %": r2_delta,
				"Classical RMSE": float(classical_metrics["rmse"]),
				"MG RMSE": float(mg_metrics["rmse"]),
				"RMSE Improvement %": rmse_delta,
				"R2 Improvement Label": r2_label,
				"RMSE Improvement Label": rmse_label,
			}
		)

		details.append(
			{
				"pair": f"{classical_name} vs {mg_name}",
				"classical_name": classical_name,
				"mg_name": mg_name,
				"classical_color": str(classical_spec["color"]),
				"mg_color": str(mg_spec["color"]),
				"classical": classical_metrics,
				"mg": mg_metrics,
			}
		)

	return pd.DataFrame(rows), details


def plot_pair_fits(
	x: np.ndarray,
	y: np.ndarray,
	classical_name: str,
	mg_name: str,
	classical_color: str,
	mg_color: str,
	classical_metrics: dict[str, Any],
	mg_metrics: dict[str, Any],
):
	grid_x = dense_prediction_grid(x)
	grid_X = as_2d(grid_x)

	fig, ax = plt.subplots(figsize=(8.8, 4.6))
	fig.patch.set_facecolor("#07111f")
	ax.set_facecolor("#07111f")

	ax.scatter(x, y, s=78, color="#d8f2ff", alpha=0.95, edgecolors="white", linewidths=0.5, label="Data", zorder=4)

	classical_y_grid = np.asarray(classical_metrics["model"].predict(grid_X), dtype=float).reshape(-1)
	mg_y_grid = np.asarray(mg_metrics["model"].predict(grid_X), dtype=float).reshape(-1)

	ax.plot(
		grid_x,
		classical_y_grid,
		"--",
		color=classical_color,
		linewidth=2.4,
		label=f"{classical_name} (R2={float(classical_metrics['r2']):.3f}, RMSE={float(classical_metrics['rmse']):.3f})",
	)
	ax.plot(
		grid_x,
		mg_y_grid,
		"-",
		color=mg_color,
		linewidth=2.6,
		label=f"{mg_name} (R2={float(mg_metrics['r2']):.3f}, RMSE={float(mg_metrics['rmse']):.3f})",
	)

	ax.set_title(f"{classical_name} vs {mg_name}", color="#e6edf7")
	ax.set_xlabel("X", color="#9db0c9")
	ax.set_ylabel("Y", color="#9db0c9")
	ax.tick_params(colors="#9db0c9")
	ax.grid(alpha=0.12)
	for spine in ax.spines.values():
		spine.set_edgecolor("#23405f")
	ax.legend(facecolor="#101c32", edgecolor="#244063", labelcolor="#e6edf7", fontsize=8.5)
	return fig


def aggregate_summary(results: dict[int, pd.DataFrame]) -> pd.DataFrame:
	model_names = results[1]["Model"].tolist()
	rows = []
	for model_name in model_names:
		per_dataset = []
		for ds_id, frame in results.items():
			per_dataset.append(frame[frame["Model"] == model_name].iloc[0])
		rows.append(
			{
				"Model": model_name,
				"Mean R2": float(np.mean([r["r2"] for r in per_dataset])),
				"Mean RMSE": float(np.mean([r["rmse"] for r in per_dataset])),
				"Mean MAE": float(np.mean([r["mae"] for r in per_dataset])),
				"Mean Rank": float(np.mean([r["Overall Rank"] for r in per_dataset])),
			}
		)
	return pd.DataFrame(rows).sort_values(["Mean Rank", "Mean RMSE"], ascending=[True, True]).reset_index(drop=True)


def leaderboard_figure(df: pd.DataFrame, metric: str, title: str, color: str):
	order = df.sort_values(metric, ascending=(metric == "Mean RMSE"))
	fig, ax = plt.subplots(figsize=(10, 4.8))
	fig.patch.set_facecolor("#07111f")
	ax.set_facecolor("#07111f")
	ax.barh(order["Model"], order[metric], color=color, alpha=0.92)
	ax.set_title(title, color="#e6edf7", pad=12)
	ax.set_xlabel(metric.replace("_", " "), color="#9db0c9")
	ax.tick_params(colors="#9db0c9")
	ax.grid(axis="x", color="white", alpha=0.08)
	for spine in ax.spines.values():
		spine.set_edgecolor("#23405f")
	return fig


def main() -> None:
	data = load_anscombe()

	with st.sidebar:
		st.markdown("## Regression Benchmark")
		st.markdown("Simple benchmark: classical vs MG models")
		st.markdown("---")

		dataset_id = st.selectbox("Dataset", [1, 2, 3, 4], format_func=lambda d: DS_NAMES[d], index=0)
		run_clicked = st.button("Run Benchmark", use_container_width=True, type="primary")

		with st.expander("Additional options", expanded=False):
			poly_degree = st.slider("Polynomial degree", min_value=2, max_value=6, value=3, step=1)
			rf_estimators = st.slider("Random Forest trees", min_value=50, max_value=500, value=200, step=25)
			rf_depth = st.slider("Random Forest depth", min_value=1, max_value=10, value=4, step=1)
			xgb_estimators = st.slider("XGBoost trees", min_value=25, max_value=400, value=150, step=25)
			xgb_depth = st.slider("XGBoost depth", min_value=1, max_value=8, value=3, step=1)
			random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
			show_weights = st.checkbox("Show MG weight plots", value=False)

		if not XGBOOST_AVAILABLE:
			st.warning("xgboost is not available in this environment.")

	if run_clicked:
		st.session_state["reg_benchmark_cfg"] = {
			"dataset_id": dataset_id,
			"poly_degree": poly_degree,
			"rf_estimators": rf_estimators,
			"rf_depth": rf_depth,
			"xgb_estimators": xgb_estimators,
			"xgb_depth": xgb_depth,
			"random_state": int(random_state),
			"show_weights": show_weights,
		}

	if "reg_benchmark_cfg" not in st.session_state:
		st.markdown(
			"""
			<div class="hero">
				<h1>Regression Benchmark for the Anscombe Quartet</h1>
				<p>Select a dataset, optionally adjust the knobs, then click <b>Run Benchmark</b>.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)
		st.info("Use the sidebar to run the benchmark.")
		return

	cfg = st.session_state["reg_benchmark_cfg"]
	dataset_id = int(cfg["dataset_id"])
	poly_degree = int(cfg["poly_degree"])
	rf_estimators = int(cfg["rf_estimators"])
	rf_depth = int(cfg["rf_depth"])
	xgb_estimators = int(cfg["xgb_estimators"])
	xgb_depth = int(cfg["xgb_depth"])
	random_state = int(cfg["random_state"])
	show_weights = bool(cfg["show_weights"])

	model_specs = build_model_specs(
		poly_degree=poly_degree,
		rf_estimators=rf_estimators,
		rf_depth=rf_depth,
		xgb_estimators=xgb_estimators,
		xgb_depth=xgb_depth,
		random_state=random_state,
	)

	x = data[dataset_id]["x"]
	y = data[dataset_id]["y"]
	current = evaluate_dataset(x, y, model_specs)
	pair_df, pair_details = evaluate_pairwise(x, y, model_specs, poly_degree)

	st.markdown(
		"""
		<div class="hero">
			<h1>Regression Benchmark for the Anscombe Quartet</h1>
			<p>
				Simple comparison between classical regressors and their MG counterparts on one selected dataset.
				Each pair is run independently and shown with separate metrics and fit plots.
			</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	st.markdown(
		f"""
		<div class="callout">
		Selected dataset: <b>{DS_NAMES[dataset_id]}</b>. KPI cards below summarize how much MG improves vs classical models.
		</div>
		""",
		unsafe_allow_html=True,
	)

	top_row = st.columns(4)

	if pair_df.empty:
		mg_better_rmse = 0
		avg_rmse_improvement = 0.0
		avg_r2_improvement = 0.0
		best_pair_label = "-"
	else:
		mg_better_rmse = int((pair_df["RMSE Improvement %"] > 0).sum())
		avg_rmse_improvement = float(pair_df["RMSE Improvement %"].mean())
		avg_r2_improvement = float(pair_df["R2 Improvement %"].mean())
		best_pair_row = pair_df.sort_values("RMSE Improvement %", ascending=False).iloc[0]
		best_pair_label = str(best_pair_row["Pair"])

	with top_row[0]:
		st.markdown(
			f"""
			<div class="kpi">
				<div class="kpi-label">MG Better RMSE (Pairs)</div>
				<div class="kpi-value">{mg_better_rmse}/{len(pair_df)}</div>
				<div class="kpi-sub">Positive means MG has lower RMSE</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with top_row[1]:
		st.markdown(
			f"""
			<div class="kpi">
				<div class="kpi-label">Avg MG RMSE Improvement</div>
				<div class="kpi-value">{avg_rmse_improvement:+.1f}%</div>
				<div class="kpi-sub">Higher is better for MG</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with top_row[2]:
		st.markdown(
			f"""
			<div class="kpi">
				<div class="kpi-label">Avg MG R2 Improvement</div>
				<div class="kpi-value">{avg_r2_improvement:+.1f}%</div>
				<div class="kpi-sub">R2 gain vs paired classical</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with top_row[3]:
		st.markdown(
			f"""
			<div class="kpi">
				<div class="kpi-label">Best MG Pair</div>
				<div class="kpi-value">{best_pair_label}</div>
				<div class="kpi-sub">Highest RMSE improvement by MG</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

	st.markdown('<div class="section-title">Dataset preview</div>', unsafe_allow_html=True)
	preview_cols = st.columns([1.1, 1])
	with preview_cols[0]:
		st.dataframe(pd.DataFrame({"X": x, "Y": y}).round(4), use_container_width=True, hide_index=True, height=300)
	with preview_cols[1]:
		fig_preview = plot_dataset_overview(x, y, f"{DS_NAMES[dataset_id]} data geometry")
		st.pyplot(fig_preview, use_container_width=True)
		plt.close(fig_preview)

	st.markdown('<div class="section-title">Benchmark table</div>', unsafe_allow_html=True)
	table = current[["Model", "Family", "r2", "rmse", "mae", "corr", "robr2", "R2 Rank", "RMSE Rank", "Overall Rank"]].copy()
	table = table.rename(columns={"r2": "R2", "rmse": "RMSE", "mae": "MAE", "corr": "Correlation", "robr2": "RobR2"})
	for col in ["R2", "RMSE", "MAE", "Correlation", "RobR2"]:
		table[col] = table[col].map(lambda v: format_float(float(v)) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "-")
	st.dataframe(table, use_container_width=True, hide_index=True)

	st.markdown('<div class="section-title">Pairwise MG vs Classical Comparison</div>', unsafe_allow_html=True)
	if pair_df.empty:
		st.warning("No pairwise MG-vs-classical models available with current options.")
	else:
		show_pairs = pair_df[
			[
				"Pair",
				"Classical R2",
				"MG R2",
				"R2 Improvement Label",
				"Classical RMSE",
				"MG RMSE",
				"RMSE Improvement Label",
			]
		].copy()
		st.dataframe(show_pairs, use_container_width=True, hide_index=True)

		pair_tabs = st.tabs([str(item["pair"]) for item in pair_details])
		for tab, detail in zip(pair_tabs, pair_details):
			with tab:
				metrics_table = pd.DataFrame(
					[
						{
							"Metric": "R2",
							"Classical": format_float(float(detail["classical"]["r2"])),
							"MG": format_float(float(detail["mg"]["r2"])),
							"Improvement %": f"{pct_change(float(detail['classical']['r2']), float(detail['mg']['r2']))[0]:+.1f}%",
						},
						{
							"Metric": "RMSE",
							"Classical": format_float(float(detail["classical"]["rmse"])),
							"MG": format_float(float(detail["mg"]["rmse"])),
							"Improvement %": f"{pct_change(float(detail['classical']['rmse']), float(detail['mg']['rmse']), lower_is_better=True)[0]:+.1f}%",
						},
					]
				)
				st.dataframe(metrics_table, use_container_width=True, hide_index=True)

				fig_pair = plot_pair_fits(
					x=x,
					y=y,
					classical_name=str(detail["classical_name"]),
					mg_name=str(detail["mg_name"]),
					classical_color=str(detail["classical_color"]),
					mg_color=str(detail["mg_color"]),
					classical_metrics=detail["classical"],
					mg_metrics=detail["mg"],
				)
				st.pyplot(fig_pair, use_container_width=True)
				plt.close(fig_pair)

	if show_weights:
		st.markdown('<div class="section-title">Machine Gnostics weights</div>', unsafe_allow_html=True)
		fig_weights = plot_weights(x, y, current, f"{DS_NAMES[dataset_id]} adaptive weights")
		st.pyplot(fig_weights, use_container_width=True)
		plt.close(fig_weights)

	st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
	st.markdown(
		"""
		<div class="callout">
		For each classical model, the paired MG model is re-run and compared directly. Use RMSE Improvement % as the main
		benchmark (positive means MG is better), then validate with R2 Improvement % and the pairwise fit plots.
		</div>
		""",
		unsafe_allow_html=True,
	)


if __name__ == "__main__":
	main()
