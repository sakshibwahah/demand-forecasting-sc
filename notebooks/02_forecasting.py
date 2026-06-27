import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

train = pd.read_csv('../data/train.csv')
features = pd.read_csv('../data/features.csv')
stores = pd.read_csv('../data/stores.csv')

df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
df = df.merge(stores, on='Store', how='left')
df['Date'] = pd.to_datetime(df['Date'])

df = df[df['Weekly_Sales'] > 0]

store1_dept1 = df[(df['Store'] == 1) & (df['Dept'] == 1)][['Date', 'Weekly_Sales']].copy()
store1_dept1.columns = ['ds', 'y']
store1_dept1 = store1_dept1.sort_values('ds').reset_index(drop=True)

split_index = int(len(store1_dept1) * 0.8)
train_df = store1_dept1.iloc[:split_index]
test_df = store1_dept1.iloc[split_index:]
split_date = train_df['ds'].max()

model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(train_df)

future = model.make_future_dataframe(periods=len(test_df), freq='W')
forecast = model.predict(future)

forecast_trimmed = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_trimmed['ds'] = pd.to_datetime(forecast_trimmed['ds']).dt.normalize()
test_df['ds'] = pd.to_datetime(test_df['ds']).dt.normalize()

test_forecast = forecast_trimmed.merge(test_df, on='ds', how='inner')

if len(test_forecast) == 0:
    forecast_trimmed['ds_week'] = forecast_trimmed['ds'].dt.to_period('W')
    test_df['ds_week'] = test_df['ds'].dt.to_period('W')
    test_forecast = forecast_trimmed.merge(test_df, on='ds_week', how='inner')
    test_forecast['ds'] = test_forecast['ds_x']

mae = mean_absolute_error(test_forecast['y'], test_forecast['yhat'])
rmse = np.sqrt(mean_squared_error(test_forecast['y'], test_forecast['yhat']))
mape = (abs(test_forecast['y'] - test_forecast['yhat']) / test_forecast['y']).mean() * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

os.makedirs('../outputs', exist_ok=True)

plt.figure(figsize=(14, 5))
plt.plot(train_df['ds'], train_df['y'], color='steelblue', label='Actual (Train)')
plt.plot(test_forecast['ds'], test_forecast['y'], color='green', label='Actual (Test)')
plt.plot(test_forecast['ds'], test_forecast['yhat'], color='red', linestyle='--', label='Forecast')
plt.fill_between(test_forecast['ds'], test_forecast['yhat_lower'], test_forecast['yhat_upper'],
                 alpha=0.2, color='red', label='Confidence Interval')
plt.title('Demand Forecast - Store 1, Dept 1')
plt.xlabel('Date')
plt.ylabel('Weekly Sales (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/forecast_store1_dept1.png')
plt.show()

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('../outputs/forecast_output.csv', index=False)
print("Forecast saved to outputs/forecast_output.csv")