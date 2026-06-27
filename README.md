# Demand Forecasting & Inventory Optimisation

An end-to-end supply chain analytics project combining demand forecasting with inventory optimisation on Walmart's retail sales data.

## Business Problem
Retail supply chains suffer from two costly extremes — overstocking (high holding costs) and stockouts (lost sales). This project uses historical sales data to forecast future demand and determine optimal inventory policies.

## Project Structure
demand-forecasting-sc/

├── data/                   # Walmart sales dataset

├── notebooks/

│   ├── 01_eda.ipynb        # Exploratory data analysis

│   ├── 02_forecasting.ipynb # Prophet demand forecasting model

│   └── 03_inventory.ipynb  # EOQ, safety stock, ROP calculations

└── outputs/                # Charts and result CSVs


## Methodology

### 1. Demand Forecasting
- Built a Facebook Prophet time series model on weekly sales data across 45 Walmart stores
- 80/20 train-test split on 3 years of historical data
- **MAPE: 10.87%** — within industry-standard accuracy for retail forecasting

### 2. Inventory Optimisation
- Fed Prophet forecasts into EOQ (Economic Order Quantity) model
- Calculated Safety Stock and Reorder Points across three service levels (90%, 95%, 99%)
- Compared EOQ-optimal policy against arbitrary 4-week ordering cycle
- **Result: 68.53% reduction in annual inventory cost**

## Key Results

| Metric | Value |
|---|---|
| Forecast MAPE | 10.87% |
| Average Weekly Demand | 19,950 units |
| EOQ | 12,884 units |
| Annual Inventory Cost (Arbitrary Policy) | $255,885 |
| Annual Inventory Cost (EOQ Optimal) | $80,523 |
| Cost Reduction | 68.53% |

## Safety Stock Analysis

| Service Level | Safety Stock | Reorder Point |
|---|---|---|
| 90% | 9,025 units | 48,927 units |
| 95% | 11,634 units | 51,536 units |
| 99% | 16,429 units | 56,331 units |

## Tech Stack
Python, Prophet, Pandas, NumPy, Scikit-learn, Matplotlib, Tableau

## Dataset
[Walmart Sales Forecast — Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)