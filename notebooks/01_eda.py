import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

train = pd.read_csv('../data/train.csv')
features = pd.read_csv('../data/features.csv')
stores = pd.read_csv('../data/stores.csv')

df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
df = df.merge(stores, on='Store', how='left')
df['Date'] = pd.to_datetime(df['Date'])

print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df.describe())

os.makedirs('../outputs', exist_ok=True)

weekly_total = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
weekly_total['Weekly_Sales_M'] = weekly_total['Weekly_Sales'] / 1e6

plt.figure(figsize=(14, 5))
plt.plot(weekly_total['Date'], weekly_total['Weekly_Sales_M'], color='steelblue', linewidth=1.5)
plt.title('Total Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales (USD millions)')
plt.tight_layout()
plt.savefig('../outputs/sales_over_time.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(df['Weekly_Sales'], bins=50, color='steelblue', edgecolor='black')
plt.title('Weekly Sales Distribution')
plt.xlabel('Weekly Sales')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('../outputs/sales_distribution.png')
plt.show()