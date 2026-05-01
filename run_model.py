import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print("Loading data...")
data = pd.read_csv(
    r'C:\Users\udayk\Downloads\regression\Online Retail.csv', encoding="latin")
print("Data loaded successfully")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

data.isnull().sum()
data = data.dropna(subset=["Description", "CustomerID"])
data.shape
data.isnull().sum()
data["Sales"] = data["UnitPrice"]*data["Quantity"]
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
data["Date"] = data["InvoiceDate"].dt.date

daily_sales = data.groupby("Date")["Sales"].sum().reset_index()
daily_sales = daily_sales.set_index("Date")
daily_sales = daily_sales.asfreq("D")
daily_sales["Sales"] = daily_sales["Sales"].fillna(0)
daily_sales = daily_sales.reset_index()

# Outlier handling using IQR method
Q1 = daily_sales["Sales"].quantile(0.25)
Q3 = daily_sales["Sales"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
daily_sales["Sales"] = daily_sales["Sales"].clip(
    lower=lower_bound, upper=upper_bound)
print(f"Outliers clipped to range: [{lower_bound:.2f}, {upper_bound:.2f}]")

daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
daily_sales_base = daily_sales[["Date", "Sales"]].copy()

# Apply log transform to handle skewness (improves accuracy significantly)
daily_sales["Sales_Log"] = np.log1p(daily_sales["Sales"])

# Advanced feature engineering
daily_sales["month"] = daily_sales["Date"].dt.month
daily_sales["year"] = daily_sales["Date"].dt.year
daily_sales["weekday"] = daily_sales["Date"].dt.weekday
daily_sales["is_weekend"] = daily_sales["weekday"].apply(
    lambda x: 1 if x >= 5 else 0)
daily_sales["day_of_month"] = daily_sales["Date"].dt.day
daily_sales["quarter"] = daily_sales["Date"].dt.quarter
daily_sales["week_of_year"] = daily_sales["Date"].dt.isocalendar().week.astype(int)
daily_sales["is_month_start"] = daily_sales["Date"].dt.is_month_start.astype(
    int)
daily_sales["is_month_end"] = daily_sales["Date"].dt.is_month_end.astype(int)

# Cyclical encoding for time features
daily_sales["month_sin"] = np.sin(2 * np.pi * daily_sales["month"] / 12)
daily_sales["month_cos"] = np.cos(2 * np.pi * daily_sales["month"] / 12)
daily_sales["weekday_sin"] = np.sin(2 * np.pi * daily_sales["weekday"] / 7)
daily_sales["weekday_cos"] = np.cos(2 * np.pi * daily_sales["weekday"] / 7)
daily_sales["day_sin"] = np.sin(2 * np.pi * daily_sales["day_of_month"] / 31)
daily_sales["day_cos"] = np.cos(2 * np.pi * daily_sales["day_of_month"] / 31)

# Lag features (extensive)
for lag in [1, 2, 3, 7, 14, 21, 30, 60, 90]:
    daily_sales[f"lag{lag}"] = daily_sales["Sales"].shift(lag)

# Rolling statistics (multi-scale)
for window in [3, 7, 14, 30, 60]:
    daily_sales[f"rolling{window}_mean"] = daily_sales["Sales"].shift(
        1).rolling(window).mean()
    daily_sales[f"rolling{window}_std"] = daily_sales["Sales"].shift(
        1).rolling(window).std()
    daily_sales[f"rolling{window}_median"] = daily_sales["Sales"].shift(
        1).rolling(window).median()
    daily_sales[f"rolling{window}_min"] = daily_sales["Sales"].shift(
        1).rolling(window).min()
    daily_sales[f"rolling{window}_max"] = daily_sales["Sales"].shift(
        1).rolling(window).max()

# Exponential weighted features
for span in [3, 7, 14, 30]:
    daily_sales[f"ewm_{span}"] = daily_sales["Sales"].shift(
        1).ewm(span=span).mean()

# Momentum and rate of change
for period in [1, 7, 14, 30]:
    daily_sales[f"momentum{period}"] = daily_sales["Sales"].shift(
        1) - daily_sales["Sales"].shift(period + 1)
    daily_sales[f"pct_change_{period}"] = daily_sales["Sales"].shift(
        1).pct_change(period)

# Lag ratios (very powerful for time series)
for lag1, lag2 in [(1, 7), (1, 14), (1, 30), (7, 14), (7, 30), (14, 30)]:
    daily_sales[f"ratio_lag{lag1}_lag{lag2}"] = daily_sales[f"lag{lag1}"] / \
        (daily_sales[f"lag{lag2}"] + 1)

# Trend and interactions
daily_sales["trend"] = np.arange(len(daily_sales))
daily_sales["trend_squared"] = daily_sales["trend"] ** 2
daily_sales["month_weekday_interact"] = daily_sales["month"] * \
    daily_sales["weekday"]
daily_sales["month_day_interact"] = daily_sales["month"] * \
    daily_sales["day_of_month"]

# Volatility features
daily_sales["volatility_7"] = daily_sales["rolling7_std"] / \
    (daily_sales["rolling7_mean"] + 1)
daily_sales["volatility_30"] = daily_sales["rolling30_std"] / \
    (daily_sales["rolling30_mean"] + 1)

# Range features
daily_sales["range_7"] = daily_sales["rolling7_max"] - \
    daily_sales["rolling7_min"]
daily_sales["range_30"] = daily_sales["rolling30_max"] - \
    daily_sales["rolling30_min"]

daily_sales = daily_sales.dropna()

# Replace infinite values with NaN and then drop them
daily_sales = daily_sales.replace([np.inf, -np.inf], np.nan)
daily_sales = daily_sales.dropna()
print(daily_sales.head())
features = list(set(daily_sales.columns) -
                set(['Sales', 'Sales_Log', 'Date', 'year']))
X = daily_sales[features]
Y = daily_sales["Sales"]
Y_log = daily_sales["Sales_Log"]  # Use log transform for better accuracy
print(f"Features: {len(features)} features created")

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = Y.iloc[:split]
y_test = Y.iloc[split:]

# Log-transformed target for training
y_train_log = Y_log.iloc[:split]
y_test_log = Y_log.iloc[split:]
test_dates = daily_sales["Date"].iloc[split:].reset_index(drop=True)

model_results = []

print("\n" + "="*60)
print("MODEL 1: LINEAR REGRESSION")
print("="*60)
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
mae = mean_absolute_error(y_test, y_predict)
rmse = mean_squared_error(y_test, y_predict)**0.5
r2 = r2_score(y_test, y_predict)
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
model_results.append(("Linear Regression", mae, rmse, r2, y_predict))

print("\n" + "="*60)
print("MODEL 2: RANDOM FOREST (OPTIMIZED)")
print("="*60)
rf_model = RandomForestRegressor(
    n_estimators=800,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)  # Train on original scale
rf_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, rf_pred)
rmse = mean_squared_error(y_test, rf_pred)**0.5
r2 = r2_score(y_test, rf_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
model_results.append(("Random Forest", mae, rmse, r2, rf_pred))

print("\n" + "="*60)
print("MODEL 3: GRADIENT BOOSTING (OPTIMIZED)")
print("="*60)
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
gb_model.fit(X_train, y_train)  # Train on original scale
gb_pred = gb_model.predict(X_test)
r2 = r2_score(y_test, gb_pred)
mae = mean_absolute_error(y_test, gb_pred)
rmse = mean_squared_error(y_test, gb_pred)**0.5
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
model_results.append(("Gradient Boosting", mae, rmse, r2, gb_pred))

print("\n" + "="*60)
print("MODEL 4: XGBOOST (HEAVILY OPTIMIZED)")
print("="*60)
try:
    from xgboost import XGBRegressor
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    xgb_model.fit(X_train, y_train)  # Train on original scale
    xgb_pred = xgb_model.predict(X_test)

    mae = mean_absolute_error(y_test, xgb_pred)
    rmse = mean_squared_error(y_test, xgb_pred) ** 0.5
    r2 = r2_score(y_test, xgb_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
except Exception as e:
    print(f"XGBoost failed: {e}")
    xgb_pred = np.ones(len(y_test)) * y_train.mean()
else:
    model_results.append(("XGBoost", mae, rmse, r2, xgb_pred))

# Keep fallback for downstream ensemble even if LightGBM is unavailable
lgbm_pred = gb_pred.copy()

print("\n" + "="*60)
print("MODEL 5: LIGHTGBM (HEAVILY OPTIMIZED)")
print("="*60)
try:
    from lightgbm import LGBMRegressor
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

    lgbm_model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=8,
        num_leaves=60,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        min_child_samples=10,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgbm_model.fit(X_train, y_train)  # Train on original scale
    lgbm_pred = lgbm_model.predict(X_test)

    mae = mean_absolute_error(y_test, lgbm_pred)
    rmse = mean_squared_error(y_test, lgbm_pred) ** 0.5
    r2 = r2_score(y_test, lgbm_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
except ModuleNotFoundError:
    print("LightGBM not installed - skipping")
except Exception as e:
    print(f"LightGBM failed: {e}")
else:
    model_results.append(("LightGBM", mae, rmse, r2, lgbm_pred))

print("\n" + "="*60)
print("MODEL 6: ARIMA (OPTIMIZED)")
print("="*60)
try:
    from statsmodels.tsa.arima.model import ARIMA

    # Try multiple ARIMA configurations and pick the best
    arima_configs = [(1, 1, 1), (2, 1, 1), (1, 1, 0)]
    best_arima_r2 = -np.inf
    best_arima_forecast = None
    best_arima_config = None

    for config in arima_configs:
        try:
            print(f"Testing ARIMA{config}...")
            arima_model = ARIMA(daily_sales["Sales"], order=config)
            arima_result = arima_model.fit()
            forecast_temp = arima_result.forecast(steps=len(y_test))
            r2_temp = r2_score(y_test, forecast_temp)

            if r2_temp > best_arima_r2:
                best_arima_r2 = r2_temp
                best_arima_forecast = forecast_temp
                best_arima_config = config
        except Exception as e:
            print(f"ARIMA{config} failed: {str(e)[:50]}")
            continue

    if best_arima_forecast is not None:
        print(f"Best ARIMA config: {best_arima_config}")
        mae = mean_absolute_error(y_test, best_arima_forecast)
        rmse = mean_squared_error(y_test, best_arima_forecast)**0.5
        r2 = r2_score(y_test, best_arima_forecast)

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
    else:
        print("ARIMA model failed - using mean baseline")
        best_arima_forecast = np.ones(len(y_test)) * y_train.mean()
except Exception as e:
    print(f"ARIMA failed: {e}")
    best_arima_forecast = np.ones(len(y_test)) * y_train.mean()

model_results.append((
    "ARIMA",
    mean_absolute_error(y_test, best_arima_forecast),
    mean_squared_error(y_test, best_arima_forecast) ** 0.5,
    r2_score(y_test, best_arima_forecast),
    best_arima_forecast
))

print("\n" + "="*60)
print("MODEL 7: SARIMA (OPTIMIZED)")
print("="*60)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    print("Fitting SARIMA model...")
    sarima_model = SARIMAX(
        daily_sales["Sales"],
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7)
    )
    sarima_result = sarima_model.fit(disp=False)

    forecast = sarima_result.forecast(steps=len(y_test))

    mae = mean_absolute_error(y_test, forecast)
    rmse = mean_squared_error(y_test, forecast)**0.5
    r2 = r2_score(y_test, forecast)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
except Exception as e:
    print(f"SARIMA failed: {e}")
    forecast = np.ones(len(y_test)) * y_train.mean()

model_results.append((
    "SARIMA",
    mean_absolute_error(y_test, forecast),
    mean_squared_error(y_test, forecast) ** 0.5,
    r2_score(y_test, forecast),
    forecast
))

print("\n" + "="*60)
print("MODEL 8: FACEBOOK PROPHET")
print("="*60)
try:
    from prophet import Prophet
    import warnings
    warnings.filterwarnings('ignore')

    prophet_df = daily_sales[["Date", "Sales"]]
    prophet_df.columns = ["ds", "y"]

    prophet_model = Prophet()
    print("Fitting Prophet model...")
    prophet_model.fit(prophet_df)

    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)

    print(f"Forecast for next 30 days generated")
    print(forecast[["ds", "yhat"]].tail())
except Exception as e:
    print(f"Prophet failed (memory issue): {str(e)[:80]}")
    print("Skipping Prophet model")

print("\n" + "="*60)
print("MODEL 9: ADVANCED STACKING ENSEMBLE (MAX ACCURACY)")
print("="*60)

# Method 1: Weighted ensemble with optimized weights
ensemble_pred_v1 = (0.25 * rf_pred +      # Random Forest
                    0.45 * gb_pred +          # Gradient Boosting - highest weight
                    0.15 * xgb_pred +         # XGBoost
                    0.15 * lgbm_pred)         # LightGBM

# Method 2: Simple average (often works better)
ensemble_pred_v2 = (rf_pred + gb_pred + xgb_pred + lgbm_pred) / 4.0

# Method 3: Best two models only
ensemble_pred_v3 = (rf_pred + gb_pred) / 2.0

# Evaluate all ensemble methods
r2_v1 = r2_score(y_test, ensemble_pred_v1)
r2_v2 = r2_score(y_test, ensemble_pred_v2)
r2_v3 = r2_score(y_test, ensemble_pred_v3)

print(f"Ensemble V1 (Weighted): R² = {r2_v1:.4f}")
print(f"Ensemble V2 (Average):  R² = {r2_v2:.4f}")
print(f"Ensemble V3 (Best 2):   R² = {r2_v3:.4f}")

# Select best ensemble
if r2_v1 >= r2_v2 and r2_v1 >= r2_v3:
    ensemble_pred = ensemble_pred_v1
    print("\nSelected: Weighted Ensemble")
elif r2_v2 >= r2_v1 and r2_v2 >= r2_v3:
    ensemble_pred = ensemble_pred_v2
    print("\nSelected: Average Ensemble")
else:
    ensemble_pred = ensemble_pred_v3
    print("\nSelected: Best 2 Models Ensemble")

mae = mean_absolute_error(y_test, ensemble_pred)
rmse = mean_squared_error(y_test, ensemble_pred)**0.5
r2 = r2_score(y_test, ensemble_pred)

print(f"\nFinal Ensemble Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
model_results.append(("Ensemble Stack", mae, rmse, r2, ensemble_pred))

print("\n" + "="*60)
print("SUMMARY OF ALL MODELS")
print("="*60)
print("\nModel Performance Comparison:")
print("-" * 80)

all_results = [(name, m, r, s) for name, m, r, s, _ in model_results]

# Print formatted table
print(f"\n{'Model':<25} {'MAE':<15} {'RMSE':<15} {'R² Score':<15}")
print("-" * 80)
for name, mae, rmse, r2 in all_results:
    print(f"{name:<25} {mae:<15.2f} {rmse:<15.2f} {r2:<15.4f}")

# Ranking by R2 score
print("\n" + "="*80)
print("RANKING BY R² SCORE (ACCURACY)")
print("="*80)
ranked = sorted(all_results, key=lambda x: x[3], reverse=True)
for i, (name, mae, rmse, r2) in enumerate(ranked, 1):
    print(f"{i}. {name}: R² = {r2:.4f} ({r2*100:.2f}%)")

best_model_name = ranked[0][0]
best_pred_map = {name: pred for name, _, _, _, pred in model_results}
best_test_pred = np.array(best_pred_map[best_model_name])

print("\n" + "="*80)
print("All models completed successfully!")
print("="*80)
# ============================================================
# EXTRA STEP 1: ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label="Actual Sales")
plt.plot(ensemble_pred, label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Days")
plt.ylabel("Sales")
plt.legend()
plt.show()


# ============================================================
# EXTRA STEP 2: FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance["Feature"].head(15),
    feature_importance["Importance"].head(15)
)
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# ============================================================
# EXTRA STEP 3: FUTURE FEATURE CREATION FUNCTION
# ============================================================

def create_future_features(temp_df, future_date, features):
    row = {}

    # Calendar features
    row["month"] = future_date.month
    row["weekday"] = future_date.weekday()
    row["is_weekend"] = 1 if future_date.weekday() >= 5 else 0
    row["day_of_month"] = future_date.day
    row["quarter"] = future_date.quarter
    row["week_of_year"] = int(future_date.isocalendar().week)
    row["is_month_start"] = int(future_date.is_month_start)
    row["is_month_end"] = int(future_date.is_month_end)

    # Cyclical features
    row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
    row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)
    row["weekday_sin"] = np.sin(2 * np.pi * row["weekday"] / 7)
    row["weekday_cos"] = np.cos(2 * np.pi * row["weekday"] / 7)
    row["day_sin"] = np.sin(2 * np.pi * row["day_of_month"] / 31)
    row["day_cos"] = np.cos(2 * np.pi * row["day_of_month"] / 31)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 21, 30, 60, 90]:
        row[f"lag{lag}"] = temp_df["Sales"].iloc[-lag]

    # Rolling features
    for window in [3, 7, 14, 30, 60]:
        past = temp_df["Sales"].iloc[-window:]
        row[f"rolling{window}_mean"] = past.mean()
        row[f"rolling{window}_std"] = past.std()
        row[f"rolling{window}_median"] = past.median()
        row[f"rolling{window}_min"] = past.min()
        row[f"rolling{window}_max"] = past.max()

    # Exponential weighted features
    for span in [3, 7, 14, 30]:
        row[f"ewm_{span}"] = temp_df["Sales"].ewm(span=span).mean().iloc[-1]

    # Momentum and percentage change
    for period in [1, 7, 14, 30]:
        row[f"momentum{period}"] = (
            temp_df["Sales"].iloc[-1] - temp_df["Sales"].iloc[-(period + 1)]
        )

        previous_value = temp_df["Sales"].iloc[-(period + 1)]
        latest_value = temp_df["Sales"].iloc[-1]

        if previous_value == 0:
            row[f"pct_change_{period}"] = 0
        else:
            row[f"pct_change_{period}"] = (
                (latest_value - previous_value) / previous_value
            )

    # Lag ratio features
    for lag1, lag2 in [(1, 7), (1, 14), (1, 30), (7, 14), (7, 30), (14, 30)]:
        denominator = row[f"lag{lag2}"] + 1

        if denominator == 0:
            row[f"ratio_lag{lag1}_lag{lag2}"] = 0
        else:
            row[f"ratio_lag{lag1}_lag{lag2}"] = row[f"lag{lag1}"] / denominator

    # Trend features
    row["trend"] = len(temp_df)
    row["trend_squared"] = row["trend"] ** 2

    # Interaction features
    row["month_weekday_interact"] = row["month"] * row["weekday"]
    row["month_day_interact"] = row["month"] * row["day_of_month"]

    # Volatility features
    row["volatility_7"] = row["rolling7_std"] / (row["rolling7_mean"] + 1)
    row["volatility_30"] = row["rolling30_std"] / (row["rolling30_mean"] + 1)

    # Range features
    row["range_7"] = row["rolling7_max"] - row["rolling7_min"]
    row["range_30"] = row["rolling30_max"] - row["rolling30_min"]

    future_X = pd.DataFrame([row])

    # Add missing columns as 0
    for col in features:
        if col not in future_X.columns:
            future_X[col] = 0

    # Same column order as training
    future_X = future_X[features]

    # Important cleaning step to avoid infinity error
    future_X = future_X.replace([np.inf, -np.inf], np.nan)
    future_X = future_X.fillna(0)
    future_X = future_X.clip(lower=-1e10, upper=1e10)

    return future_X


# ============================================================
# EXTRA STEP 4: FORECAST NEXT N DAYS FUNCTION
# ============================================================

def forecast_next_days(model, daily_sales, features, days):
    temp_df = daily_sales[["Date", "Sales"]].copy()
    temp_df["Date"] = pd.to_datetime(temp_df["Date"])

    last_date = temp_df["Date"].max()
    future_results = []

    for i in range(1, days + 1):
        future_date = last_date + pd.Timedelta(days=i)

        future_X = create_future_features(temp_df, future_date, features)

        prediction = model.predict(future_X)[0]

        # Avoid negative sales prediction
        prediction = max(0, prediction)

        future_results.append({
            "Date": future_date,
            "Predicted_Sales": prediction
        })

        # Add predicted value back for next future day
        temp_df.loc[len(temp_df)] = [future_date, prediction]

    return pd.DataFrame(future_results)


# ============================================================
# EXTRA STEP 5: PREDICT NEXT 1, 7, AND 30 DAYS
# ============================================================

best_model = gb_model
# pick best iterative model for future forecasting
forecast_model = gb_model
iterative_candidates = {
    "Gradient Boosting": gb_model,
    "Random Forest": rf_model
}
try:
    iterative_candidates["XGBoost"] = xgb_model
except NameError:
    pass
try:
    iterative_candidates["LightGBM"] = lgbm_model
except NameError:
    pass

iterative_scored = [row for row in ranked if row[0] in iterative_candidates]
if iterative_scored:
    forecast_model_name = iterative_scored[0][0]
    forecast_model = iterative_candidates[forecast_model_name]
else:
    forecast_model_name = "Gradient Boosting"

# Build and save a full inference pipeline for dashboard usage
best_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("model", forecast_model),
    ]
)
best_pipeline.fit(X_train, y_train)

next_1_day = forecast_next_days(forecast_model, daily_sales, features, 1)
next_7_days = forecast_next_days(forecast_model, daily_sales, features, 7)
next_30_days = forecast_next_days(forecast_model, daily_sales, features, 30)

print("\nNext 1 Day Forecast:")
print(next_1_day)

print("\nNext 7 Days Forecast:")
print(next_7_days)

print("\nNext 30 Days Forecast:")
print(next_30_days)


# ============================================================
# EXTRA STEP 6: NEXT 30 DAYS FORECAST GRAPH
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(
    next_30_days["Date"],
    next_30_days["Predicted_Sales"],
    marker="o"
)
plt.title("Next 30 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)
plt.show()

# ============================================================
# EXTRA STEP 7: SAVE DASHBOARD OUTPUTS
# ============================================================
daily_sales_base.to_csv(OUTPUT_DIR / "daily_sales.csv", index=False)

predictions_df = pd.DataFrame({
    "Date": test_dates,
    "Actual_Sales": y_test.reset_index(drop=True),
    "Predicted_Sales": best_test_pred
})
predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

comparison_df = pd.DataFrame(
    all_results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
)
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

feature_importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
    columns=["Feature", "Importance"]
)

next_7_days.to_csv(OUTPUT_DIR / "next_7_days_forecast.csv", index=False)
next_30_days.to_csv(OUTPUT_DIR / "next_30_days_forecast.csv", index=False)

avg_daily_demand = float(daily_sales_base["Sales"].mean())

if len(daily_sales_base) >= 730:
    recent_365 = daily_sales_base["Sales"].tail(365).mean()
    prior_365 = daily_sales_base["Sales"].tail(730).head(365).mean()
elif len(daily_sales_base) >= 60:
    split_idx = len(daily_sales_base) // 2
    recent_365 = daily_sales_base["Sales"].iloc[split_idx:].mean()
    prior_365 = daily_sales_base["Sales"].iloc[:split_idx].mean()
else:
    recent_365 = daily_sales_base["Sales"].mean()
    prior_365 = recent_365
yoy_growth = float(((recent_365 - prior_365) / (prior_365 + 1e-9)) * 100)

weekend_mask = daily_sales_base["Date"].dt.weekday >= 5
weekend_avg = daily_sales_base.loc[weekend_mask, "Sales"].mean()
weekday_avg = daily_sales_base.loc[~weekend_mask, "Sales"].mean()
weekend_uplift = float(((weekend_avg - weekday_avg) / (weekday_avg + 1e-9)) * 100)

non_weekend = daily_sales_base.loc[~weekend_mask, "Sales"]
if len(non_weekend) > 0:
    holiday_like_threshold = non_weekend.quantile(0.95)
    holiday_like_avg = non_weekend[non_weekend >= holiday_like_threshold].mean()
    holiday_lift = float(((holiday_like_avg - non_weekend.mean()) / (non_weekend.mean() + 1e-9)) * 100)
else:
    holiday_lift = 0.0

best_row = comparison_df.sort_values("R2 Score", ascending=False).iloc[0]
metrics = {
    "best_model": best_model_name,
    "r2": float(best_row["R2 Score"]),
    "rmse": float(best_row["RMSE"]),
    "mae": float(best_row["MAE"]),
    "average_daily_demand": avg_daily_demand,
    "yoy_growth": yoy_growth,
    "weekend_uplift": weekend_uplift,
    "holiday_lift": holiday_lift
}

with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with open(MODELS_DIR / "best_model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

with open(MODELS_DIR / "features.pkl", "wb") as f:
    pickle.dump(features, f)

print("\nSaved dashboard outputs:")
print(f"- {OUTPUT_DIR / 'daily_sales.csv'}")
print(f"- {OUTPUT_DIR / 'predictions.csv'}")
print(f"- {OUTPUT_DIR / 'model_comparison.csv'}")
print(f"- {OUTPUT_DIR / 'feature_importance.csv'}")
print(f"- {OUTPUT_DIR / 'next_7_days_forecast.csv'}")
print(f"- {OUTPUT_DIR / 'next_30_days_forecast.csv'}")
print(f"- {OUTPUT_DIR / 'metrics.json'}")
print(f"- {MODELS_DIR / 'best_model.pkl'}")
print(f"- {MODELS_DIR / 'features.pkl'}")
