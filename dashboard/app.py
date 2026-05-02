import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title="AI-Driven Sales Forecasting & Decision Intelligence Platform", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    .stApp {
        background:
            radial-gradient(circle at 20% 10%, rgba(46,232,143,0.14), rgba(7,11,16,0) 35%),
            radial-gradient(circle at 85% 15%, rgba(57,201,255,0.12), rgba(7,11,16,0) 36%),
            radial-gradient(circle at 50% 55%, rgba(46,232,143,0.10), rgba(7,11,16,0) 42%),
            #070b10;
        color: #f4f7fb;
    }
    .block-container {
        max-width: 1380px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    * { color: inherit; }
    p, span, div, li, label, h1, h2, h3, h4, h5 {
        color: #f4f7fb !important;
    }
    .muted { color: #94a3b8 !important; }
    .badge-pill {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        border: 1px solid #1b2733;
        background: rgba(16,22,28,0.9);
        color: #2ee88f !important;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.6px;
    }
    .main-title {
        margin-top: 12px;
        font-size: 48px;
        line-height: 1.1;
        font-weight: 900;
        color: #f4f7fb !important;
    }
    .subtitle {
        margin-top: 10px;
        font-size: 15px;
        line-height: 1.5;
        color: #94a3b8 !important;
        max-width: 900px;
    }
    .panel {
        background: #10161c;
        border: 1px solid #1b2733;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 18px 30px rgba(0,0,0,0.35);
    }
    .refresh-title {
        font-size: 13px;
        color: #94a3b8 !important;
        margin-bottom: 8px;
    }
    .refresh-value {
        font-size: 17px;
        font-weight: 800;
        color: #f4f7fb !important;
    }
    .kpi-title {
        font-size: 12px;
        letter-spacing: 0.8px;
        font-weight: 800;
        color: #94a3b8 !important;
        margin-bottom: 10px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 900;
        color: #f4f7fb !important;
        line-height: 1;
    }
    .kpi-sub {
        margin-top: 10px;
        color: #2ee88f !important;
        font-size: 13px;
    }
    .section-title {
        font-size: 28px;
        font-weight: 900;
        margin-bottom: 10px;
        color: #f4f7fb !important;
    }
    .metric-box {
        background: #0d141a;
        border: 1px solid #1b2733;
        border-radius: 14px;
        padding: 12px 14px;
        height: 110px;
    }
    .metric-label {
        color: #94a3b8 !important;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f4f7fb !important;
        font-size: 28px;
        font-weight: 900;
    }
    .metric-desc {
        color: #94a3b8 !important;
        font-size: 12px;
        margin-top: 4px;
    }
    .info-card-title {
        font-size: 18px;
        font-weight: 800;
        color: #f4f7fb !important;
        margin-bottom: 8px;
    }
    .info-card-text {
        color: #94a3b8 !important;
        font-size: 14px;
        line-height: 1.5;
    }
    .progress-wrap { margin-top: 10px; }
    .progress-row { margin-bottom: 14px; }
    .progress-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 13px;
        color: #f4f7fb !important;
    }
    .progress-track {
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: #1a2330;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #39c9ff, #2ee88f);
    }
    div[data-testid="stDataFrame"] {
        background: #10161c;
        border: 1px solid #1b2733;
        border-radius: 14px;
        padding: 6px;
    }
    thead tr th { color: #2ee88f !important; }
    tbody tr td { color: #f4f7fb !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = Path(BASE_DIR) / "outputs"
MODELS_DIR = Path(BASE_DIR) / "models"


def fmt_inr(value: float, decimals: int = 2) -> str:
    """Display-only rupee formatting (no unit conversion)."""
    return f"₹{value:,.{decimals}f}"


def read_csv_or_none(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_json_or_none(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


daily_sales = read_csv_or_none(OUTPUT_DIR / "daily_sales.csv")
if daily_sales is None or not {"Date", "Sales"}.issubset(daily_sales.columns):
    st.error("Missing or invalid `outputs/daily_sales.csv`. Run `run_model.py` to generate outputs from your dataset.")
    st.stop()

daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
daily_sales["Sales"] = pd.to_numeric(daily_sales["Sales"], errors="coerce").fillna(0.0)
daily_sales = daily_sales.sort_values("Date").reset_index(drop=True)

model_comp = read_csv_or_none(OUTPUT_DIR / "model_comparison.csv")
metrics_json = read_json_or_none(OUTPUT_DIR / "metrics.json")

if model_comp is None or not {"Model", "MAE", "RMSE", "R2 Score"}.issubset(model_comp.columns):
    if metrics_json and metrics_json.get("best_model") is not None:
        model_comp = pd.DataFrame(
            [
                {
                    "Model": str(metrics_json.get("best_model", "")),
                    "MAE": float(metrics_json.get("mae", 0) or 0),
                    "RMSE": float(metrics_json.get("rmse", 0) or 0),
                    "R2 Score": float(metrics_json.get("r2", 0) or 0),
                }
            ]
        )
    else:
        st.error("Missing `outputs/model_comparison.csv` (or `outputs/metrics.json` with best_model, mae, rmse, r2). Run `run_model.py`.")
        st.stop()

feature_importance = read_csv_or_none(OUTPUT_DIR / "feature_importance.csv")
if feature_importance is None or not {"Feature", "Importance"}.issubset(feature_importance.columns):
    feature_importance = pd.DataFrame(columns=["Feature", "Importance"])
    st.warning("`outputs/feature_importance.csv` missing or invalid — feature bars will be empty until you run `run_model.py`.")

model_comp = model_comp.copy()
model_comp["R2 Score"] = pd.to_numeric(model_comp["R2 Score"], errors="coerce").fillna(-1e9)
model_comp["MAE"] = pd.to_numeric(model_comp["MAE"], errors="coerce").fillna(0)
model_comp["RMSE"] = pd.to_numeric(model_comp["RMSE"], errors="coerce").fillna(0)
# loaded from outputs/model_comparison.csv (or fallback row built from metrics.json)
best_row = model_comp.sort_values("R2 Score", ascending=False).iloc[0]
best_model_name = str(best_row["Model"])
best_r2 = float(best_row["R2 Score"])
best_mae = float(best_row["MAE"])
best_rmse = float(best_row["RMSE"])
ranked_models = model_comp.sort_values("R2 Score", ascending=False).reset_index(drop=True)


def load_model_artifacts():
    model_path = MODELS_DIR / "best_model.pkl"
    features_path = MODELS_DIR / "features.pkl"
    if not model_path.exists() or not features_path.exists():
        return None, None, False
    try:
        with open(model_path, "rb") as f:
            model_obj = pickle.load(f)
    except Exception as e:
        st.error("Model loading failed. Please retrain model with the same scikit-learn version.")
        st.exception(e)
        model_obj = None
        return None, None, False
    try:
        with open(features_path, "rb") as f:
            feature_cols = pickle.load(f)
        target_is_log = False
        # Backward compatibility with older saved dict format
        if isinstance(model_obj, dict) and "pipeline" in model_obj:
            target_is_log = bool(model_obj.get("target_is_log", False))
            model_obj = model_obj["pipeline"]
            if not feature_cols and "feature_columns" in model_obj:
                feature_cols = model_obj["feature_columns"]
        return model_obj, list(feature_cols), target_is_log
    except Exception:
        return None, None, False


def create_future_features(temp_df, future_date, feature_cols):
    row = {}
    future_date = pd.Timestamp(future_date)
    hist = temp_df.copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist["Sales"] = pd.to_numeric(hist["Sales"], errors="coerce").fillna(0)

    row["month"] = future_date.month
    row["weekday"] = future_date.weekday()
    row["is_weekend"] = 1 if row["weekday"] >= 5 else 0
    row["day_of_month"] = future_date.day
    row["quarter"] = future_date.quarter
    row["week_of_year"] = int(future_date.isocalendar().week)
    row["is_month_start"] = int(future_date.is_month_start)
    row["is_month_end"] = int(future_date.is_month_end)

    row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
    row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)
    row["weekday_sin"] = np.sin(2 * np.pi * row["weekday"] / 7)
    row["weekday_cos"] = np.cos(2 * np.pi * row["weekday"] / 7)
    row["day_sin"] = np.sin(2 * np.pi * row["day_of_month"] / 31)
    row["day_cos"] = np.cos(2 * np.pi * row["day_of_month"] / 31)

    sales_series = hist["Sales"].reset_index(drop=True)

    def safe_lag(lag):
        if len(sales_series) >= lag:
            return float(sales_series.iloc[-lag])
        return float(sales_series.iloc[-1]) if len(sales_series) else 0.0

    for lag in [1, 2, 3, 7, 14, 21, 30, 60, 90]:
        row[f"lag{lag}"] = safe_lag(lag)

    for window in [3, 7, 14, 30, 60]:
        past = sales_series.iloc[-window:] if len(sales_series) >= window else sales_series
        row[f"rolling{window}_mean"] = float(past.mean()) if len(past) else 0.0
        row[f"rolling{window}_std"] = float(past.std()) if len(past) > 1 else 0.0
        row[f"rolling{window}_median"] = float(past.median()) if len(past) else 0.0
        row[f"rolling{window}_min"] = float(past.min()) if len(past) else 0.0
        row[f"rolling{window}_max"] = float(past.max()) if len(past) else 0.0

    for span in [3, 7, 14, 30]:
        row[f"ewm_{span}"] = float(sales_series.ewm(span=span).mean().iloc[-1]) if len(sales_series) else 0.0

    for period in [1, 7, 14, 30]:
        current = safe_lag(1)
        previous = safe_lag(period + 1)
        row[f"momentum{period}"] = current - previous
        row[f"pct_change_{period}"] = 0.0 if previous == 0 else (current - previous) / previous

    for lag1, lag2 in [(1, 7), (1, 14), (1, 30), (7, 14), (7, 30), (14, 30)]:
        denom = row[f"lag{lag2}"] + 1
        row[f"ratio_lag{lag1}_lag{lag2}"] = 0.0 if denom == 0 else row[f"lag{lag1}"] / denom

    row["trend"] = len(hist)
    row["trend_squared"] = row["trend"] ** 2
    row["month_weekday_interact"] = row["month"] * row["weekday"]
    row["month_day_interact"] = row["month"] * row["day_of_month"]

    row["volatility_7"] = row["rolling7_std"] / (row["rolling7_mean"] + 1)
    row["volatility_30"] = row["rolling30_std"] / (row["rolling30_mean"] + 1)
    row["range_7"] = row["rolling7_max"] - row["rolling7_min"]
    row["range_30"] = row["rolling30_max"] - row["rolling30_min"]

    row_df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return row_df


def forecast_next_days(model_obj, base_daily_sales, feature_cols, days, target_is_log=False):
    temp_df = base_daily_sales[["Date", "Sales"]].copy()
    temp_df["Date"] = pd.to_datetime(temp_df["Date"])
    temp_df["Sales"] = pd.to_numeric(temp_df["Sales"], errors="coerce").fillna(0)
    last_date = temp_df["Date"].max()
    results = []
    for i in range(1, days + 1):
        future_date = last_date + pd.Timedelta(days=i)
        X_future = create_future_features(temp_df, future_date, feature_cols)
        pred = float(model_obj.predict(X_future)[0])
        if target_is_log:
            pred = float(np.expm1(pred))
        pred = max(0.0, pred)
        results.append({"Date": future_date, "Predicted_Sales": pred})
        temp_df.loc[len(temp_df)] = [future_date, pred]
    return pd.DataFrame(results)


def generate_recent_predictions(model_obj, base_daily_sales, feature_cols, window=30, target_is_log=False):
    ds = base_daily_sales[["Date", "Sales"]].copy()
    ds["Date"] = pd.to_datetime(ds["Date"])
    ds = ds.sort_values("Date").reset_index(drop=True)
    if len(ds) < 2:
        return pd.DataFrame(columns=["Date", "Actual_Sales", "Predicted_Sales"])
    min_history = 91
    max_window = len(ds) - min_history
    if max_window < 1:
        window = len(ds) - 1
        start_idx = 1
    else:
        window = min(window, max_window)
        window = max(1, window)
        start_idx = len(ds) - window
    history_df = ds.iloc[:start_idx].copy()
    eval_df = ds.iloc[start_idx:].copy()
    preds = []
    for _, row in eval_df.iterrows():
        X_eval = create_future_features(history_df, row["Date"], feature_cols)
        pred_raw = float(model_obj.predict(X_eval)[0])
        if target_is_log:
            pred_raw = float(np.expm1(pred_raw))
        pred = max(0.0, pred_raw)
        preds.append(pred)
        history_df.loc[len(history_df)] = [row["Date"], row["Sales"]]
    return pd.DataFrame(
        {
            "Date": eval_df["Date"].values,
            "Actual_Sales": eval_df["Sales"].values,
            "Predicted_Sales": preds,
        }
    )


model_obj, feature_columns, target_is_log = load_model_artifacts()

predictions = read_csv_or_none(OUTPUT_DIR / "predictions.csv")
if predictions is not None and {"Date", "Actual_Sales", "Predicted_Sales"}.issubset(predictions.columns):
    predictions = predictions.copy()
else:
    predictions = None

if predictions is None:
    if model_obj is None or not feature_columns:
        st.error(
            "Missing `outputs/predictions.csv` and trained model (`models/best_model.pkl` + `models/features.pkl`). "
            "Run `run_model.py` to generate them."
        )
        st.stop()
    predictions = generate_recent_predictions(
        model_obj, daily_sales, feature_columns, window=32, target_is_log=target_is_log
    )

if predictions is None or len(predictions) == 0:
    st.error("Could not load or build predictions. Run `run_model.py`.")
    st.stop()

next_7_dynamic = read_csv_or_none(OUTPUT_DIR / "next_7_days_forecast.csv")
next_30_dynamic = read_csv_or_none(OUTPUT_DIR / "next_30_days_forecast.csv")

if next_7_dynamic is None or not {"Date", "Predicted_Sales"}.issubset(next_7_dynamic.columns):
    if model_obj is None or not feature_columns:
        st.error("Missing `outputs/next_7_days_forecast.csv`. Run `run_model.py` or ensure the model artifacts exist.")
        st.stop()
    next_7_dynamic = forecast_next_days(
        model_obj, daily_sales, feature_columns, 7, target_is_log=target_is_log
    )

if next_30_dynamic is None or not {"Date", "Predicted_Sales"}.issubset(next_30_dynamic.columns):
    if model_obj is None or not feature_columns:
        st.error("Missing `outputs/next_30_days_forecast.csv`. Run `run_model.py` or ensure the model artifacts exist.")
        st.stop()
    next_30_dynamic = forecast_next_days(
        model_obj, daily_sales, feature_columns, 30, target_is_log=target_is_log
    )

features = feature_columns if feature_columns else []

predictions["Date"] = pd.to_datetime(predictions["Date"])
predictions["Actual_Sales"] = pd.to_numeric(predictions["Actual_Sales"], errors="coerce").fillna(0.0)
predictions["Predicted_Sales"] = pd.to_numeric(predictions["Predicted_Sales"], errors="coerce").fillna(0.0)

next_7_dynamic["Date"] = pd.to_datetime(next_7_dynamic["Date"])
next_7_dynamic["Predicted_Sales"] = pd.to_numeric(next_7_dynamic["Predicted_Sales"], errors="coerce").fillna(0.0)
next_30_dynamic["Date"] = pd.to_datetime(next_30_dynamic["Date"])
next_30_dynamic["Predicted_Sales"] = pd.to_numeric(next_30_dynamic["Predicted_Sales"], errors="coerce").fillna(0.0)

forecast_horizon_days = int(len(next_30_dynamic))

if model_obj is None or not feature_columns:
    st.info(
        "Trained model files not found under `models/`. The dashboard uses `outputs/*.csv` for charts and forecasts. "
        "Run `run_model.py` to create `best_model.pkl` for custom-date predictions."
    )

# Header
left, right = st.columns([4.6, 1.4], gap="large")
with left:
    st.markdown('<div class="badge-pill">✦ FORECAST STUDIO · V1.1</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">AI-Driven Sales Forecasting &amp; Decision Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-time forecasting, model selection, and business-ready insights powered by machine learning</div>',
        unsafe_allow_html=True,
    )
with right:
    last_refresh_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(
        f"""
        <div class="panel">
            <div class="refresh-title">Last refresh</div>
            <div class="refresh-value">{last_refresh_utc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Runtime dependency warning (keeps app from crashing)
if not PLOTLY_AVAILABLE:
    st.warning(
        "Plotly is not installed in this Python environment. "
        "Install with: `python -m pip install plotly`"
    )

# KPI row
# calculated from dataset
avg_daily_demand = float(daily_sales["Sales"].mean())
# calculated from dataset
weekend_avg = daily_sales[daily_sales["Date"].dt.weekday >= 5]["Sales"].mean()
# calculated from dataset
weekday_avg = daily_sales[daily_sales["Date"].dt.weekday < 5]["Sales"].mean()
# calculated from dataset
weekend_uplift = float(((weekend_avg - weekday_avg) / (weekday_avg + 1e-9)) * 100)
# calculated from dataset
first_30_avg = daily_sales["Sales"].head(30).mean()
if len(daily_sales) >= 60:
    # calculated from dataset
    mid_30_avg = daily_sales["Sales"].iloc[-60:-30].mean()
else:
    # calculated from dataset
    mid_30_avg = daily_sales["Sales"].head(30).mean()
# calculated from dataset
last_30_avg = daily_sales["Sales"].tail(30).mean()
# calculated from dataset (last 30 days vs first 30 days average)
growth = float(((last_30_avg - first_30_avg) / (first_30_avg + 1e-9)) * 100)
# calculated from dataset
trend = float(((last_30_avg - mid_30_avg) / (mid_30_avg + 1e-9)) * 100)

k1, k2, k3, k4 = st.columns(4)
cards = [
    ("AVG DAILY DEMAND", fmt_inr(avg_daily_demand), f"{growth:+.1f}% vs first 30 days"),
    ("GROWTH", f"{growth:+.1f}%", "First 30 vs last 30 days"),
    ("WEEKEND UPLIFT", f"{weekend_uplift:+.1f}%", "vs weekday baseline"),
    ("FORECAST HORIZON", f"{forecast_horizon_days} Days", "Saved 30-day forecast length"),
]
for col, (title, value, sub) in zip([k1, k2, k3, k4], cards):
    with col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# Actual vs Predicted (Plotly graph_objects only)
# calculated from dataset (residual spread of predictions vs actuals)
residual_std = (predictions["Actual_Sales"] - predictions["Predicted_Sales"]).std()
pred_upper = predictions["Predicted_Sales"] + residual_std
pred_lower = predictions["Predicted_Sales"] - residual_std

st.markdown('<div class="panel"><div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
if PLOTLY_AVAILABLE:
    fig_main = go.Figure()
    fig_main.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=pred_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=pred_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(57,201,255,0.18)",
            name="Confidence Band",
            hoverinfo="skip",
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Actual_Sales"],
            mode="lines",
            name="Actual",
            line=dict(color="#2ee88f", width=3),
            hovertemplate="₹%{y:,.2f}<extra></extra>",
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Predicted_Sales"],
            mode="lines",
            name="Predicted",
            line=dict(color="#39c9ff", width=3),
            hovertemplate="₹%{y:,.2f}<extra></extra>",
        )
    )
    fig_main.update_layout(
        template=None,
        height=470,
        paper_bgcolor="#10161c",
        plot_bgcolor="#10161c",
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(color="#f4f7fb"),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color="#f4f7fb")),
        xaxis=dict(gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.2)",
            tickfont=dict(color="#94a3b8"),
            title="Sales (₹)",
            tickprefix="₹",
            tickformat=",.0f",
        ),
    )
    st.plotly_chart(fig_main, use_container_width=True)
else:
    st.info("Install Plotly to render this chart.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Model performance section — R² / RMSE / MAE below: loaded from outputs/model_comparison.csv (see best_row above)
# calculated from dataset (unless MAPE provided in metrics.json)
mae = (predictions["Actual_Sales"] - predictions["Predicted_Sales"]).abs().mean()
rmse = ((predictions["Actual_Sales"] - predictions["Predicted_Sales"]) ** 2).mean() ** 0.5
if metrics_json is not None and metrics_json.get("mape") is not None:
    mape = float(metrics_json["mape"])
else:
    valid = predictions["Actual_Sales"] > 0
    if valid.sum() > 0:
        # calculated from dataset
        mape = (
            (predictions.loc[valid, "Actual_Sales"] - predictions.loc[valid, "Predicted_Sales"]).abs()
            / predictions.loc[valid, "Actual_Sales"]
        ).mean() * 100
    else:
        mape = 0.0

st.markdown('<div class="panel"><div class="section-title">Model Performance</div>', unsafe_allow_html=True)
mm1, mm2, mm3, mm4 = st.columns(4)
metrics = [
    ("Best Model", best_model_name, "Top model from model_comparison.csv"),
    ("R² Score", f"{best_r2:.3f}", "Best model R² score"),
    ("RMSE", fmt_inr(best_rmse, 0), "Best model RMSE"),
    ("MAE", fmt_inr(best_mae, 0), "Best model MAE"),
]
for col, (lab, val, desc) in zip([mm1, mm2, mm3, mm4], metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">{lab}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Accuracy highlight (from best model R²)
if best_r2 >= 0.80:
    acc_level = "High Accuracy"
    acc_note = "Model performance is strong and suitable for operational decisions."
elif best_r2 >= 0.60:
    acc_level = "Moderate Accuracy"
    acc_note = "Model is useful but should be monitored for volatile periods."
elif best_r2 >= 0.40:
    acc_level = "Developing Accuracy"
    acc_note = "Model needs refinement for better stability."
else:
    acc_level = "Low Accuracy"
    acc_note = "Model quality is currently low; review features and retraining strategy."

st.markdown(
    f"""
    <div style="margin-top:12px; padding:12px 14px; border-radius:12px; border:1px solid #1b2733; background:#0d141a;">
        <div style="font-size:12px; color:#94a3b8;">Forecast Quality</div>
        <div style="font-size:20px; font-weight:900; color:#2ee88f; margin-top:4px;">{acc_level}</div>
        <div style="font-size:13px; color:#94a3b8; margin-top:4px;">Best Model: {best_model_name}</div>
        <div style="font-size:13px; color:#94a3b8; margin-top:4px;">{acc_note}</div>
        <div style="font-size:13px; color:#94a3b8; margin-top:4px;">MAPE (valid actuals only): {mape:.2f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Key drivers of sale (below model performance)
weekend_mask = daily_sales["Date"].dt.weekday >= 5
# calculated from dataset
weekday_avg = daily_sales.loc[~weekend_mask, "Sales"].mean()
# calculated from dataset
weekend_avg = daily_sales.loc[weekend_mask, "Sales"].mean()
# calculated from dataset
seasonality_delta = ((weekend_avg - weekday_avg) / (weekday_avg + 1e-9)) * 100
# calculated from dataset
volatility_cv = (daily_sales["Sales"].std() / (daily_sales["Sales"].mean() + 1e-9)) * 100
head_n = min(30, len(daily_sales))
# calculated from dataset
start_avg = daily_sales["Sales"].head(head_n).mean()
# calculated from dataset
end_avg = daily_sales["Sales"].tail(head_n).mean()
trend_delta = trend

st.markdown(
    '<div style="margin-top:16px; margin-bottom:8px; font-size:20px; font-weight:800; color:#f4f7fb;">Key Drivers of Sale</div>',
    unsafe_allow_html=True,
)
kds1, kds2, kds3 = st.columns(3)
with kds1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Seasonality</div>
            <div class="metric-value">{seasonality_delta:+.1f}%</div>
            <div class="metric-desc">Weekend vs weekday demand effect</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kds2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Demand Volatility</div>
            <div class="metric-value">{volatility_cv:.1f}%</div>
            <div class="metric-desc">Coefficient of variation in daily sales</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kds3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Trend Analysis</div>
            <div class="metric-value">{trend_delta:+.1f}%</div>
            <div class="metric-desc">Recent period vs early period average</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Dynamic model explanation
model_explanations = {
    "Linear Regression": "Linear Regression is simple, interpretable, and works well when the relationship between features and sales is mostly linear.",
    "Random Forest": "Random Forest performs well because it combines many decision trees, reduces overfitting, and handles non-linear relationships in sales data.",
    "Gradient Boosting": "Gradient Boosting performs well because it learns sequentially from previous errors and captures complex non-linear patterns in lag and seasonal features.",
    "XGBoost": "XGBoost is powerful for tabular data because it uses optimized gradient boosting, regularization, and handles complex feature interactions.",
    "LightGBM": "LightGBM is efficient and accurate on large tabular datasets because it trains faster and handles many engineered features well.",
    "ARIMA": "ARIMA is useful for pure time-series forecasting because it models trend and autocorrelation in historical sales values.",
    "SARIMA": "SARIMA improves ARIMA by adding seasonal patterns, making it useful when weekly or monthly seasonality exists.",
    "Ensemble Stack": "Ensemble performs well because it combines strengths of multiple models and reduces individual model weaknesses.",
}

if len(ranked_models) > 1:
    second_row = ranked_models.iloc[1]
    second_model = str(second_row["Model"])
    second_r2 = float(second_row["R2 Score"])
    second_rmse = float(second_row["RMSE"])
else:
    second_model = best_model_name
    second_r2 = best_r2
    second_rmse = best_rmse

r2_diff = best_r2 - second_r2
rmse_diff = second_rmse - best_rmse
best_description = model_explanations.get(
    best_model_name,
    "This model was selected because it achieved the best evaluation score among all compared models.",
)

st.markdown(
    f"""
    <div class="panel">
        <div class="section-title">Why {best_model_name} is the Best Model</div>
        <div class="info-card-text">
            {best_model_name} is selected as the best model because it achieved the highest R² score of {best_r2:.3f}.
            It outperformed {second_model} by +{r2_diff:.3f} R² and reduced RMSE by {fmt_inr(rmse_diff, 0)}.
        </div>
        <div class="info-card-text" style="margin-top:10px;">
            {best_description}
        </div>
        <div class="info-card-text" style="margin-top:12px;">
            <b style="color:#f4f7fb;">Model Selection Rule:</b>
            The model with the highest R² score is selected as the best model. If R² scores are close,
            RMSE and MAE are checked to choose the model with lower prediction error.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Real-time custom prediction from pickle model
st.markdown('<div class="panel"><div class="section-title">Predict Sales for a Custom Date</div>', unsafe_allow_html=True)
if model_obj is None or not feature_columns:
    st.warning("Run run_model.py locally and push updated best_model.pkl.")
else:
    last_date = daily_sales["Date"].max()
    st.write("Last Date in Dataset:", last_date)
    default_date = last_date + pd.Timedelta(days=1)
    cdate, cbtn = st.columns([2, 1], gap="large")
    with cdate:
        custom_date = st.date_input(
            "Select date",
            value=default_date.date(),
            min_value=default_date.date(),
        )
    with cbtn:
        predict_clicked = st.button("Predict Sales", use_container_width=True)

    if predict_clicked:
        temp_hist = daily_sales.copy()
        selected = pd.Timestamp(custom_date)
        if selected <= last_date:
            st.warning("Please select a future date")
        else:
            future_X = create_future_features(temp_hist, selected, features)
            future_X = future_X[features]
            pred_value = float(model_obj.predict(future_X)[0])
            if target_is_log:
                pred_value = np.expm1(pred_value)
            pred_value = max(0.0, pred_value)

            st.markdown(
                f"""
                <div style="margin-top:10px;" class="metric-box">
                    <div class="metric-label">Predicted Sales for {selected.date()}</div>
                    <div class="metric-value">{fmt_inr(pred_value)}</div>
                    <div class="metric-desc">Real-time inference from models/best_model.pkl</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # calculated from dataset (low vs historical sales distribution)
            low_pred_threshold = float(daily_sales["Sales"].quantile(0.01))
            if pred_value < low_pred_threshold:
                st.warning("Prediction unusually low relative to historical daily sales (below ~1st percentile); check feature generation.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Next 7 Days Forecast table + chart
st.markdown('<div class="panel"><div class="section-title">Next 7 Days Forecast</div>', unsafe_allow_html=True)
next_7_display = next_7_dynamic.copy()
next_7_display["Predicted_Sales"] = next_7_display["Predicted_Sales"].map(lambda x: fmt_inr(float(x)))
st.dataframe(next_7_display, use_container_width=True, hide_index=True)
if PLOTLY_AVAILABLE:
    fig_7 = go.Figure()
    fig_7.add_trace(
        go.Scatter(
            x=next_7_dynamic["Date"],
            y=next_7_dynamic["Predicted_Sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#2ee88f", width=3),
            marker=dict(color="#2ee88f", size=7),
            hovertemplate="₹%{y:,.2f}<extra></extra>",
        )
    )
    fig_7.update_layout(
        template=None,
        height=320,
        paper_bgcolor="#10161c",
        plot_bgcolor="#10161c",
        font=dict(color="#f4f7fb"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.2)",
            tickfont=dict(color="#94a3b8"),
            title="Sales (₹)",
            tickprefix="₹",
            tickformat=",.0f",
        ),
        legend=dict(font=dict(color="#f4f7fb")),
    )
    st.plotly_chart(fig_7, use_container_width=True)
else:
    st.info("Install Plotly to render this chart.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Next 30 Days Forecast table + chart
st.markdown('<div class="panel"><div class="section-title">Next 30 Days Forecast</div>', unsafe_allow_html=True)
next_30_display = next_30_dynamic.copy()
next_30_display["Predicted_Sales"] = next_30_display["Predicted_Sales"].map(lambda x: fmt_inr(float(x)))
st.dataframe(next_30_display, use_container_width=True, hide_index=True)
if PLOTLY_AVAILABLE:
    fig_30 = go.Figure()
    fig_30.add_trace(
        go.Scatter(
            x=next_30_dynamic["Date"],
            y=next_30_dynamic["Predicted_Sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#2ee88f", width=3),
            marker=dict(color="#2ee88f", size=5),
            hovertemplate="₹%{y:,.2f}<extra></extra>",
        )
    )
    fig_30.update_layout(
        template=None,
        height=360,
        paper_bgcolor="#10161c",
        plot_bgcolor="#10161c",
        font=dict(color="#f4f7fb"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.2)",
            tickfont=dict(color="#94a3b8"),
            title="Sales (₹)",
            tickprefix="₹",
            tickformat=",.0f",
        ),
        legend=dict(font=dict(color="#f4f7fb")),
    )
    st.plotly_chart(fig_30, use_container_width=True)
else:
    st.info("Install Plotly to render this chart.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Feature section
ws_left, ws_right = st.columns(2, gap="large")

# calculated from dataset (mean sales by weekday)
weekly = daily_sales.copy()
weekly["weekday"] = weekly["Date"].dt.day_name()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekly = weekly.groupby("weekday", as_index=False)["Sales"].mean()
weekly["weekday"] = pd.Categorical(weekly["weekday"], categories=weekday_order, ordered=True)
weekly = weekly.sort_values("weekday")

with ws_left:
    st.markdown('<div class="panel"><div class="section-title">What moves the model</div><div class="muted">Weekly Seasonality</div>', unsafe_allow_html=True)
    if PLOTLY_AVAILABLE:
        fig_week = go.Figure(
            data=[
                go.Bar(
                    x=weekly["weekday"],
                    y=weekly["Sales"],
                    marker=dict(color="#2ee88f"),
                    name="Weekly Seasonality",
                    hovertemplate="₹%{y:,.2f}<extra></extra>",
                )
            ]
        )
        fig_week.update_layout(
            template=None,
            height=360,
            paper_bgcolor="#10161c",
            plot_bgcolor="#10161c",
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(color="#f4f7fb"),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(148,163,184,0.2)", tickfont=dict(color="#94a3b8")),
            yaxis=dict(
                gridcolor="rgba(148,163,184,0.2)",
                tickfont=dict(color="#94a3b8"),
                title="Avg sales (₹)",
                tickprefix="₹",
                tickformat=",.0f",
            ),
        )
        st.plotly_chart(fig_week, use_container_width=True)
    else:
        st.info("Install Plotly to render this chart.")
    st.markdown("</div>", unsafe_allow_html=True)

with ws_right:
    if len(feature_importance) == 0:
        st.markdown(
            """<div class="panel">
<div class="section-title">Feature Importance</div>
<div class="muted">No data — run `run_model.py` to generate `outputs/feature_importance.csv`.</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        fi_top = feature_importance.sort_values("Importance", ascending=False).head(8).copy()
        fi_max = fi_top["Importance"].max() if len(fi_top) else 1
        fi_top["pct"] = (fi_top["Importance"] / (fi_max + 1e-9) * 100).clip(lower=2)

        rows = []
        for _, row in fi_top.iterrows():
            rows.append(
                f"""<div class="progress-row">
<div class="progress-head">
<span>{row['Feature']}</span>
<span>{row['Importance']:.3f}</span>
</div>
<div class="progress-track">
<div class="progress-fill" style="width:{row['pct']:.1f}%;"></div>
</div>
</div>"""
            )
        st.markdown(
            f"""<div class="panel">
<div class="section-title">Feature Importance</div>
<div class="muted">Ranked contribution weights using horizontal progress bars.</div>
<div class="progress-wrap">
{''.join(rows)}
</div>
</div>""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Business insights
st.markdown('<div class="section-title">Business Insights</div>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3, gap="large")
_roll_w = max(2, min(7, len(daily_sales)))
_roll_std = daily_sales["Sales"].rolling(_roll_w, min_periods=2).std()
_roll_mean = daily_sales["Sales"].rolling(_roll_w, min_periods=2).mean()
_rolling_cv_pct = (_roll_std / (_roll_mean + 1e-9) * 100).replace([np.inf, -np.inf], np.nan).dropna()
# calculated from dataset (compare overall CV to rolling CV distribution)
volatility_high_threshold = float(np.percentile(_rolling_cv_pct, 75)) if len(_rolling_cv_pct) else volatility_cv + 1.0
if weekend_uplift < 0:
    seasonality_text = "Weekend demand is lower than weekday baseline. Focus promotions or staffing more on weekdays."
else:
    seasonality_text = "Weekend demand is higher than weekday baseline. Increase inventory and staffing before weekends."

if volatility_cv > volatility_high_threshold:
    volatility_text = "Demand is highly volatile relative to typical rolling variability in this dataset. Use safety stock and monitor forecast errors weekly."
else:
    volatility_text = "Demand is relatively stable. Current forecast can support planning decisions."

if trend > 0:
    trend_text = "Sales trend is increasing. Prepare inventory and operations for higher demand."
else:
    trend_text = "Sales trend is decreasing. Review pricing, promotions, and demand drivers."

insights = [
    ("Seasonality", seasonality_text),
    ("Demand Volatility", volatility_text),
    ("Trend Analysis", trend_text),
]
for col, (title, text) in zip([b1, b2, b3], insights):
    with col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="info-card-title">{title}</div>
                <div class="info-card-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Recommendations 2x2
st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
r1c1, r1c2 = st.columns(2, gap="large")
r2c1, r2c2 = st.columns(2, gap="large")
rec_cards = [
    ("Inventory Management", "Increase safety buffers ahead of high-load weekday transitions and weekend demand ramps."),
    ("Pricing Strategy", "Use controlled markdown windows during softer days to smooth demand and improve utilization."),
    ("Logistics Planning", "Prioritize staffing and outbound capacity on projected surge intervals."),
    ("Performance Monitoring", "Track MAE/RMSE drift weekly and trigger retraining thresholds automatically."),
]
for col, (title, text) in zip([r1c1, r1c2, r2c1, r2c2], rec_cards):
    with col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="info-card-title">{title}</div>
                <div class="info-card-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Risk section
st.markdown('<div class="section-title">Risk</div>', unsafe_allow_html=True)
rk1, rk2 = st.columns(2, gap="large")
with rk1:
    st.markdown(
        """
        <div class="panel">
            <div class="info-card-title">Key Risks</div>
            <div class="info-card-text">
                • Promotion shock amplification<br/>
                • Holiday calendar displacement<br/>
                • Structural demand shifts by channel
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rk2:
    st.markdown(
        """
        <div class="panel">
            <div class="info-card-title">Mitigation Strategies</div>
            <div class="info-card-text">
                • Dynamic safety stock policy<br/>
                • Event-adjusted forecast overrides<br/>
                • Continuous retraining and backtesting
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
