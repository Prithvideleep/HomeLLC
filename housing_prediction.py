import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Set path for saving/loading model
model_path = 'models/housing_model-r1.pkl'

# Load and clean data
df = pd.read_csv('final_dataset.csv')
df.columns = df.columns.str.strip()
df.rename(columns=lambda x: x.strip(), inplace=True)

# Handle date
df['Date'] = pd.to_datetime(df['DATE'] if 'DATE' in df.columns else df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Fill missing values
# df = df.fillna(method='ffill').fillna(method='bfill')
df = df.ffill().bfill()
# Feature engineering
df['MortgageRate_Lag1'] = df['MortgageRate'].shift(1)
df['MortgageRate_Lag2'] = df['MortgageRate'].shift(2)
df['UnemploymentRate_Lag1'] = df['UnemploymentRate'].shift(1)
df['GDP_Lag1'] = df['GDP'].shift(1)
df['CPI_Lag1'] = df['CPI'].shift(1)
df['HousingStarts_Lag1'] = df['HousingStarts'].shift(1)
df['Mortgage_Unemployment_Interaction'] = df['MortgageRate'] * df['UnemploymentRate']
df['Time_Trend'] = np.arange(len(df))
df['Month'] = df.index.month
df['Year'] = df.index.year
df.dropna(inplace=True)

# Transform target variable
target = 'HomePriceIndex'
df['HomePriceIndex_Transformed'], lambda_val = boxcox(df[target])
target_transformed = 'HomePriceIndex_Transformed'

# Select features
features = [col for col in df.columns if col not in [target, target_transformed, 'DATE', 'Date']]
X = df[features]
y = df[target_transformed]

# Load or train model
if os.path.exists(model_path):
    print(f"\n Loading saved model from {model_path}...")
    model = joblib.load(model_path)
else:
    print("\n Training model and saving to disk...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f" Model saved to {model_path}")

# Backtesting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
y_pred_trans = model.predict(X_test)
y_pred = inv_boxcox(y_pred_trans, lambda_val)
y_test_actual = df.loc[y_test.index, target]

backtest_df = pd.DataFrame({'HPI': y_test_actual, 'Predicted': y_pred, 'Type': 'Backtest (GBR)'}, index=y_test.index)

# Forecast future using Gradient Boosting
future_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=24, freq='ME')
future_X = pd.DataFrame([X.iloc[-1].values] * 24, columns=X.columns, index=future_dates)
future_y_trans = model.predict(future_X)
future_y = inv_boxcox(future_y_trans, lambda_val)
future_df = pd.DataFrame({'HPI': future_y, 'Type': 'Future (GBR)'}, index=future_dates)

# SARIMAX Forecast
try:
    sarimax_order = (1, 1, 1)
    sarimax_model = SARIMAX(df[target_transformed], order=sarimax_order).fit(disp=False)
    sarimax_pred = sarimax_model.get_forecast(steps=24)
    sarimax_mean = inv_boxcox(sarimax_pred.predicted_mean, lambda_val)
    conf_int = sarimax_pred.conf_int()
    sarimax_df = pd.DataFrame({
        'HPI': sarimax_mean,
        'Lower_CI': inv_boxcox(conf_int.iloc[:, 0], lambda_val),
        'Upper_CI': inv_boxcox(conf_int.iloc[:, 1], lambda_val),
        'Type': 'Future (SARIMAX)'
    }, index=future_dates)
except Exception as e:
    print(f" SARIMAX error: {e}")
    sarimax_df = pd.DataFrame()

# Prophet Forecast
prophet_df = pd.DataFrame({'ds': df.index, 'y': df[target]})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
future_prophet = prophet_model.make_future_dataframe(periods=24, freq='M')
forecast_prophet = prophet_model.predict(future_prophet)
prophet_future = forecast_prophet.tail(24).set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]
prophet_df_future = pd.DataFrame({
    'HPI': prophet_future['yhat'],
    'Lower_CI': prophet_future['yhat_lower'],
    'Upper_CI': prophet_future['yhat_upper'],
    'Type': 'Future (Prophet)'
})

# Plotting
plot_df = pd.concat([
    pd.DataFrame({'HPI': df[target], 'Type': 'Actual'}, index=df.index),
    backtest_df[['Predicted']].rename(columns={'Predicted': 'HPI'}).assign(Type='Backtest (GBR)'),
    future_df,
    sarimax_df if not sarimax_df.empty else None,
    prophet_df_future if not prophet_df_future.empty else None
])

plt.figure(figsize=(18, 10))
for label, group in plot_df.groupby('Type'):
    plt.plot(group.index, group['HPI'], label=label, linewidth=2)

# CI Shading
if not sarimax_df.empty:
    plt.fill_between(sarimax_df.index, sarimax_df['Lower_CI'], sarimax_df['Upper_CI'], color='purple', alpha=0.2, label='SARIMAX CI')
if not prophet_df_future.empty:
    plt.fill_between(prophet_df_future.index, prophet_df_future['Lower_CI'], prophet_df_future['Upper_CI'], color='orange', alpha=0.2, label='Prophet CI')

plt.title('Home Price Index Forecast')
plt.xlabel('Date')
plt.ylabel('HPI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('hpi_forecast_all_models.png')
plt.show()

# --- Inference and Conclusions Section ---
# Analyze the forecast and write conclusions to a txt file
try:
    last_actual = df[target].iloc[-1]
    last_pred = future_df['HPI'].iloc[-1]
    percent_change = ((last_pred - last_actual) / last_actual) * 100

    trend = "increase" if percent_change > 0 else "decrease" if percent_change < 0 else "remain stable"
    conclusion = (
        f"Home Price Index (HPI) Forecast Inference\n"
        f"-----------------------------------------\n"
        f"Last actual HPI value: {last_actual:.2f}\n"
        f"Predicted HPI after 24 months: {last_pred:.2f}\n"
        f"Expected percent change over 24 months: {percent_change:.2f}%\n"
        f"\n"
        f"Conclusion:\n"
        f"The model forecasts an overall {trend} in the Home Price Index over the next 24 months.\n"
        f"\n"
        f"Note: This forecast is based on Gradient Boosting, SARIMAX, and Prophet models. "
        f"Confidence intervals from SARIMAX and Prophet are shown in the plot. "
        f"Interpret results with caution, as forecasts depend on input data and model assumptions."
    )

    with open("hpi_forecast_inference.txt", "w") as f:
        f.write(conclusion)
except Exception as e:
    print(f"Could not write inference file: {e}")
