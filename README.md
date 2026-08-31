# Demand Forecasting & Inventory Optimisation
### An End-to-End Supply Chain Analytics Tool

A full-stack SC analytics project built on Walmart's retail sales data. Combines ML-based demand forecasting with inventory optimisation across 21 Class A departments, deployed as an interactive Streamlit application.

**Live App:** https://demand-forecasting-sc-4z3zvtcabjysdxfblxbdem.streamlit.app/
**Dataset:** [Walmart Sales Forecast — Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

---

## Business Problem

Retail supply chains face two costly extremes — overstocking (high holding costs) and stockouts (lost sales and customer dissatisfaction). This project builds a data-driven pipeline to forecast demand accurately and determine optimal inventory policies, directly addressing the trade-off between service level and cost.

---

## Project Architecture

```
demand-forecasting-sc/
├── data/
│   ├── train.csv               # Weekly sales data (421,570 rows, 45 stores, 99 departments)
│   ├── features.csv            # External factors: temperature, fuel price, CPI, markdowns
│   └── stores.csv              # Store type and size
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_forecasting.ipynb    # Prophet vs XGBoost model comparison
│   ├── 03_inventory.ipynb      # EOQ, safety stock, ROP calculations
│   ├── 04_abc_analysis.ipynb   # ABC inventory classification
│   └── 05_multi_sku_forecast.ipynb  # Multi-SKU forecasting across 21 Class A departments
├── models/
│   └── xgb_models.pkl          # Trained XGBoost models (21 departments)
├── outputs/                    # Generated CSVs and charts
├── app.py                      # Streamlit dashboard
└── requirements.txt
```

---

## Methodology

### 1. Exploratory Data Analysis
- 3 years of weekly sales data (Feb 2010 — Oct 2012) across 45 Walmart stores and 99 departments
- Identified seasonal demand spikes during Black Friday and Christmas weeks
- Filtered negative and zero-sales weeks to ensure data quality

### 2. Demand Forecasting — Prophet vs XGBoost

Compared two forecasting approaches on Store 1, Department 1:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Facebook Prophet | 1,938.78 | 3,649.56 | 10.87% |
| **XGBoost** | **1,170.80** | **1,636.94** | **5.84%** |

XGBoost outperformed Prophet by nearly 2x on MAPE. XGBoost leveraged engineered time-series features (lag_1, lag_4, lag_52, rolling means) that captured Walmart's weekly seasonality more effectively than Prophet's additive decomposition for this structured retail dataset.

### 3. ABC Inventory Classification

Classified all 81 departments by annual revenue contribution using the standard APICS Pareto methodology (70%/90% boundaries):

| Category | Departments | Annual Sales (USD M) | Revenue Share |
|---|---|---|---|
| A — High Priority | 21 | 4,686.81 | 69.56% |
| B — Medium Priority | 17 | 1,338.55 | 19.87% |
| C — Low Priority | 43 | 711.95 | 10.57% |

Top 3 departments by revenue: Dept 92 ($483.94M), Dept 95 ($449.32M), Dept 38 ($393.12M).

### 4. Multi-SKU Forecasting — 21 Class A Departments

Scaled the XGBoost pipeline across all 21 Class A departments:

- **Average MAPE: 4.63%** across all departments
- Best performing: Dept 40 (MAPE 1.79%), Dept 8 (MAPE 2.05%)
- Highest error: Dept 5 (MAPE 15.31%) — driven by extreme holiday demand spikes

### 5. Inventory Optimisation — EOQ Model

Applied Economic Order Quantity (EOQ) optimisation to each Class A department using forecasted demand as input.

**Assumptions (Chopra & Meindl, Supply Chain Management, 2016):**
- Ordering cost: $2000/order — covers PO processing, transportation fixed cost, receiving/inspection
- Holding cost: 25% of unit cost annually — capital (~15%), storage (~5%), shrinkage (~5%)
- Service level: 95% (z = 1.65)
- Lead time: 2 weeks (typical Walmart domestic supplier replenishment)
- Baseline comparison: bi-weekly ordering cycle (common retail replenishment policy)

**Results across 21 Class A departments:**

| Metric | Value |
|---|---|
| Average Cost Reduction vs Baseline | 35.91% |
| Total EOQ Annual Cost | $6,206,493.66 |
| Total Baseline Annual Cost |$10,360,536.01 |
| **Total Annual Saving** | **$4,154,042.35** |

**Single-SKU example (Dept 1, Store 1):**

| Metric | Value |
|---|---|
| Avg Weekly Demand | 18,783.08 units |
| EOQ | 17,679.10 units |
| Safety Stock (95% SL) | 3910.0 units |
| Reorder Point | 41477.0 units |
| EOQ Annual Cost | $220,988.73 |
| Baseline Cost (2-week cycle) | $286788.55 |
| Cost Reduction | 22.94% |

---

## Streamlit Dashboard

Five interactive tabs:

**Tab 1 — Demand Forecast**
Select any Class A department and view XGBoost forecast vs actual sales with MAPE metric.

**Tab 2 — Inventory Recommendations**
EOQ, Safety Stock, and Reorder Point for selected department. Replenishment alert system with three risk zones (Reorder Now / Monitor / Safe).

**Tab 3 — Scenario Simulator**
Adjust demand change %, ordering cost, holding cost, and lead time in real time. See how EOQ and annual cost respond dynamically. Useful for Black Friday planning, supplier cost changes, or lead time disruptions.

**Tab 4 — Stockout Risk Predictor**
Enter current stock level and lead time. Returns stockout probability using the z-score method assuming normally distributed demand during lead time. Traffic light output: <10% (safe), 10–30% (monitor), >30% (critical).

**Tab 5 — ABC Classification**
Full Pareto chart and classification table across all 81 departments.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML / Forecasting | XGBoost, Facebook Prophet, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

---

## Key Findings

1. **XGBoost significantly outperforms Prophet** for structured retail time series — MAPE of 5.84% vs 10.87%, driven by lag features capturing weekly and annual seasonality
2. **21 departments (26%) drive 69.6% of revenue** — ABC analysis enables focused inventory management on high-impact SKUs
3. **EOQ policy saves $7.2M annually** across Class A departments vs a bi-weekly replenishment baseline
4. **Dept 5 shows highest forecast error (15.31%)** due to extreme holiday demand spikes — recommends higher safety stock buffer during Q4
5. **Safety stock at 95% service level** reduces stockout probability to <5% during standard 2-week lead time periods

---

## Setup & Installation

```bash
git clone https://github.com/sakshibwahah/demand-forecasting-sc.git
cd demand-forecasting-sc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Download the Walmart dataset from Kaggle and place the CSV files in the `data/` folder before running.

---

## References

-Operations Management: Sustainability and Supply Chain Management (12th Edition)
by Jay Heizer (Author), Barry Render (Author), Chuck Munson (Author) 
- Walmart Sales Forecast Dataset — Kaggle (2014).
