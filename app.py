import pandas as pd
import streamlit as st

st.set_page_config(page_title="Macroeconomic Forecasting App")
st.title("Macroeconomic Forecasting App")

import numpy as np
from xgboost import XGBRegressor
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM


def check_password():
	if st.session_state.get("authenticated"):
		return True

	st.title("Private Macroeconomic Forecasting App")
	password = st.text_input("Password", type="password")

	if st.button("Log in"):
		app_password = st.secrets.get("APP_PASSWORD")
		if not app_password:
			st.error("APP_PASSWORD is missing from .streamlit/secrets.toml")
		elif password == app_password:
			st.session_state["authenticated"] = True
			st.rerun()
		else:
			st.error("Incorrect password")

	return False


if not check_password():
	st.stop()

data = pd.read_csv("R1_model.csv").sort_values("Year")
data["Year"] = data["Year"].astype(int)

default_targets = [
	"GDP_growth",
	"Inflation_rate",
	"Real_wage",
	"Labor_productivity",
	"Exchange_rate_PHP_to_USD",
]
available_targets = [column for column in default_targets if column in data]
available_inputs = [column for column in data.columns if column != "Year"]

def safe_mape(actual, forecast):
	actual = np.asarray(actual, dtype=float)
	forecast = np.asarray(forecast, dtype=float)
	mask = actual != 0
	return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100 if mask.any() else np.nan

def mase_scale(values):
	values = np.asarray(values, dtype=float)
	differences = np.diff(values)
	return np.mean(np.abs(differences)) if len(differences) else np.nan

def metrics(actual, forecast, training):
	actual = np.asarray(actual, dtype=float)
	forecast = np.asarray(forecast, dtype=float)
	error = actual - forecast
	scale = mase_scale(training)
	return {
		"RMSE": np.sqrt(np.mean(error**2)),
		"MAE": np.mean(np.abs(error)),
		"MAPE": safe_mape(actual, forecast),
		"MASE": np.mean(np.abs(error)) / scale if scale and not np.isnan(scale) else np.nan,
	}

def fit_univariate(method, values, horizon):
	values = pd.Series(values, dtype=float).dropna().to_numpy()
	if len(values) < 12:
		raise ValueError("at least 12 observations are required")
	if method == "ARIMA":
		return ARIMA(values, order=(1, 1, 1), trend="t").fit().forecast(horizon)
	if method == "GARCH":
		model = arch_model(values, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
		fitted = model.fit(disp="off")
		return fitted.forecast(horizon=horizon, reindex=False).mean.iloc[-1].to_numpy()
	if method == "XGBoost":
		lag_count = min(3, max(1, len(values) // 8))
		features = np.asarray([
			values[index - lag_count:index]
			for index in range(lag_count, len(values))
		])
		model = XGBRegressor(
			n_estimators=200,
			max_depth=2,
			learning_rate=0.05,
			objective="reg:squarederror",
			random_state=42,
		)
		model.fit(features, values[lag_count:])
		history = list(values)
		forecasts = []
		for _ in range(horizon):
			prediction = float(model.predict(np.asarray(history[-lag_count:]).reshape(1, -1))[0])
			forecasts.append(prediction)
			history.append(prediction)
		return np.asarray(forecasts)
	raise ValueError(f"unsupported univariate method: {method}")

def fit_multivariate(method, frame, target, horizon):
	complete = frame.dropna()
	if len(complete) < 25:
		raise ValueError("at least 25 complete five-variable observations are required")
	if method == "VAR":
		model = VAR(complete).fit(maxlags=2, trend="c")
		forecast = model.forecast(complete.to_numpy()[-model.k_ar:], horizon)
		return forecast[:, complete.columns.get_loc(target)]
	if method == "VECM":
		model = VECM(complete, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
		forecast = model.predict(steps=horizon)
		return forecast[:, complete.columns.get_loc(target)]
	raise ValueError(f"unsupported multivariate method: {method}")

def backtest(method, target, series, panel, folds=5):
	if method in {"VAR", "VECM"}:
		usable = panel.dropna()
		if len(usable) < 25:
			raise ValueError("not enough complete observations for multivariate backtesting")
		horizon = 1
		first_test = max(20, len(usable) - folds)
		actual = []
		predicted = []
		training = usable[target].iloc[:first_test].to_numpy()
		for end in range(first_test, len(usable)):
			predicted.extend(fit_multivariate(method, usable.iloc[:end], target, horizon))
			actual.append(usable[target].iloc[end])
		return metrics(actual, predicted, training)

	usable = series.dropna().reset_index(drop=True)
	if len(usable) < 15:
		raise ValueError("not enough observations for backtesting")
	first_test = max(10, len(usable) - folds)
	actual = usable.iloc[first_test:].to_numpy()
	predicted = []
	for end in range(first_test, len(usable)):
		predicted.extend(fit_univariate(method, usable.iloc[:end], 1))
	return metrics(actual, predicted, usable.iloc[:first_test].to_numpy())

with st.sidebar:
	st.header("Forecast settings")
	selected_targets = st.multiselect("Forecast outputs", available_targets, default=available_targets)
	selected_inputs = st.multiselect(
		"Forecast inputs",
		available_inputs,
		default=[column for column in available_inputs if column not in available_targets],
		help="Choose variables whose future values you want to enter below.",
	)
	selected_methods = st.multiselect(
		"Methods to compare",
		["ARIMA", "GARCH", "VAR", "VECM", "XGBoost"],
		default=["ARIMA", "VAR", "VECM"],
	)
	forecast_start = 2027
	horizon = 5
	st.write("Forecast period: 2027-2031")
	run_forecast = st.button("Run comparison", type="primary", use_container_width=True)

future_years = np.arange(forecast_start, forecast_start + horizon)

st.subheader("Customizable data preview")
preview_years = [2025, 2026, 2027]
preview_data = data[data["Year"].isin(preview_years)].copy()
preview_data.index = preview_data["Year"]
preview_data = preview_data.drop(columns=["Year"])
editable_preview = st.data_editor(
	preview_data,
	hide_index=False,
	num_rows="fixed",
	use_container_width=True,
	key="customizable_preview_v2",
)
data.index = data["Year"]
data = data.drop(columns=["Year"])
data.update(editable_preview)
data = data.reset_index()
data = data.rename(columns={"index": "Year"})
st.caption("Edit values for 2025-2027. These changes are used in the forecasting run.")

st.subheader("Data quality")
quality_columns = st.columns(3)
quality_columns[0].metric("Observations", len(data))
quality_columns[1].metric("Variables", len(data.columns) - 1)
quality_columns[2].metric("Incomplete cells", int(data.isna().sum().sum()))

missingness = (
	data.isna().sum().rename("Missing values").to_frame()
	.assign(Observed=lambda frame: len(data) - frame["Missing values"])
	.assign(Coverage=lambda frame: (frame["Observed"] / len(data)).round(3))
	.sort_values("Coverage")
)
st.dataframe(missingness, use_container_width=True)

if not selected_targets:
	st.warning("Select at least one forecast output.")
elif run_forecast:
	panel = data.copy()
	panel.index = panel["Year"]
	panel = panel.drop(columns=["Year"])[available_targets]
	if selected_inputs:
		st.info("Selected inputs are available in the editable preview. ARIMA, GARCH, VAR, and VECM currently forecast from target histories.")
	results = []
	forecast_tables = []

	for target in selected_targets:
		for method in selected_methods:
			try:
				if method in {"VAR", "VECM"}:
					forecast = fit_multivariate(method, panel, target, horizon)
				else:
					forecast = fit_univariate(method, data[target], horizon)
				score = backtest(method, target, data[target], panel)
				results.append({"Target": target, "Method": method, **score, "Status": "ok"})
				forecast_tables.append(pd.DataFrame({"Year": future_years, "Target": target, "Method": method, "Forecast": forecast}))
			except Exception as error:
				results.append({"Target": target, "Method": method, "Status": str(error)})

	st.subheader("Model comparison")
	comparison = pd.DataFrame(results)
	st.dataframe(comparison, use_container_width=True)
	successful_forecasts = [table for table in forecast_tables if not table.empty]
	if successful_forecasts:
		st.subheader("Five-year forecasts")
		forecasts = pd.concat(successful_forecasts, ignore_index=True)
		forecast_matrix = forecasts.pivot_table(index="Year", columns=["Target", "Method"], values="Forecast")
		st.dataframe(forecast_matrix, use_container_width=True)
else:
	st.info("Select outputs and methods, then click Run comparison. VAR and VECM use complete rows across the selected outputs.")
