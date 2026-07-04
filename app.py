import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

st.set_page_config(page_title="SC Analytics Dashboard", page_icon="📦", layout="wide")

@st.cache_data
def load_data():
    train = pd.read_csv('data/train.csv')
    features = pd.read_csv('data/features.csv')
    stores = pd.read_csv('data/stores.csv')
    df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(stores, on='Store', how='left')
    df['Date'] = pd.to_datetime(df['Date'])
    # Consistency fix: include zero sales
    df = df[df['Weekly_Sales'] >= 0]
    return df

@st.cache_data
def load_outputs():
    abc = pd.read_csv('outputs/abc_classification.csv')
    inventory = pd.read_csv('outputs/multi_sku_inventory.csv')
    forecast_metrics = pd.read_csv('outputs/multi_sku_forecast_metrics.csv')
    return abc, inventory, forecast_metrics

@st.cache_resource
def load_models():
    with open('models/xgb_models.pkl', 'rb') as f:
        return pickle.load(f)

def create_features(df):
    df = df.copy()
    df['week'] = df['ds'].dt.isocalendar().week.astype(int)
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['year'] = df['ds'].dt.year
    df['lag_1'] = df['y'].shift(1)
    df['lag_4'] = df['y'].shift(4)
    df['lag_52'] = df['y'].shift(52)
    df['rolling_mean_4'] = df['y'].shift(1).rolling(4).mean()
    df['rolling_mean_12'] = df['y'].shift(1).rolling(12).mean()
    return df

features_cols = ['week', 'month', 'quarter', 'year', 'lag_1', 'lag_4', 'lag_52', 'rolling_mean_4', 'rolling_mean_12']

df = load_data()
abc, inventory, forecast_metrics = load_outputs()
models = load_models()
class_a_depts = abc[abc['ABC_Category'] == 'A']['Dept'].tolist()

st.title("Supply Chain Analytics Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Demand Forecast", "Inventory Recommendations", "Scenario Simulator", "Stockout Risk", "ABC Analysis"])

with tab1:
    st.header("Demand Forecast by Department")
    selected_dept = st.selectbox("Select Department", class_a_depts, key="dept_forecast")
    dept_df = df[df['Dept'] == selected_dept].groupby('Date')['Weekly_Sales'].sum().reset_index()
    dept_df.columns = ['ds', 'y']
    dept_df = dept_df.sort_values('ds').reset_index(drop=True)
    dept_feat = create_features(dept_df).dropna()
    split_idx = int(len(dept_feat) * 0.8)
    train_d, test_d = dept_feat.iloc[:split_idx], dept_feat.iloc[split_idx:]
    model = models[selected_dept]
    preds = model.predict(test_d[features_cols])
    mape = forecast_metrics[forecast_metrics['Dept'] == selected_dept]['MAPE_%'].values
    mape_val = mape[0] if len(mape) > 0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Department", f"Dept {selected_dept}")
    col2.metric("Forecast MAPE", f"{mape_val:.2f}%")
    col3.metric("ABC Category", "A — High Priority")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_d['ds'], y=train_d['y'], mode='lines', name='Actual (Train)', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=test_d['ds'], y=test_d['y'], mode='lines', name='Actual (Test)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test_d['ds'], y=preds, mode='lines', name='XGBoost Forecast', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Inventory Recommendations")
    selected_dept2 = st.selectbox("Select Department", class_a_depts, key="dept_inventory")
    inv_row = inventory[inventory['Dept'] == selected_dept2].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Weekly Demand", f"{inv_row['Avg_Weekly_Demand']:,.0f} units")
    col2.metric("EOQ", f"{inv_row['EOQ']:,.0f} units")
    col3.metric("Safety Stock (95%)", f"{inv_row['Safety_Stock_95']:,.0f} units")
    col4.metric("Reorder Point", f"{inv_row['ROP']:,.0f} units")
    
    st.subheader("Cost Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Arbitrary Policy Cost", f"${inv_row['Arbitrary_Annual_Cost']:,.2f}")
    col2.metric("EOQ Optimal Cost", f"${inv_row['EOQ_Annual_Cost']:,.2f}")
    col3.metric("Cost Reduction", f"{inv_row['Cost_Reduction_%']:.2f}%", delta=f"-{inv_row['Cost_Reduction_%']:.2f}% cost saved", delta_color="inverse")
    st.caption("EOQ vs. 2-week cycle policy.")
    
    current_stock = st.number_input("Enter current stock level (units)", min_value=0, value=int(inv_row['ROP']), step=100)
    if current_stock <= inv_row['ROP']: st.error(f"REORDER NOW — Current stock ({current_stock:,.0f}) is at or below ROP ({inv_row['ROP']:,.0f})")
    elif current_stock <= inv_row['ROP'] * 1.15: st.warning(f"MONITOR — Current stock ({current_stock:,.0f}) is within 15% of ROP ({inv_row['ROP']:,.0f})")
    else: st.success(f"SAFE — Current stock ({current_stock:,.0f}) is above ROP ({inv_row['ROP']:,.0f})")
    st.caption("ROP accounts for average demand during lead time plus safety stock at 95% service level. Monitor zone triggers at 15% above ROP.")

with tab3:
    st.header("Scenario Simulator")
    selected_dept3 = st.selectbox("Select Department", class_a_depts, key="dept_scenario")
    inv_row3 = inventory[inventory['Dept'] == selected_dept3].iloc[0]
    demand_change = st.slider("Demand Change (%)", min_value=-50, max_value=100, value=0, step=5)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: unit_cost = st.number_input("Unit Cost ($)", min_value=1.0, value=25.0, step=5.0)
    with col_b: ordering_cost = st.number_input("Ordering Cost ($)", min_value=1.0, value=500.0, step=50.0)
    with col_c: holding_cost_pct = st.number_input("Annual Holding Cost (%)", min_value=1, max_value=100, value=25)
    with col_d: lead_time = st.number_input("Lead Time (weeks)", min_value=1, value=2, step=1, key="lt_tab3")
    
    holding_cost = unit_cost * (holding_cost_pct / 100)
    new_demand = inv_row3['Avg_Weekly_Demand'] * (1 + demand_change / 100)
    new_eoq = np.sqrt((2 * (new_demand * 52) * ordering_cost) / holding_cost)
    new_ss = 1.65 * df[df['Dept'] == selected_dept3].groupby('Date')['Weekly_Sales'].sum().std() * np.sqrt(lead_time)
    new_cost = (new_eoq / 2) * holding_cost + ((new_demand * 52) / new_eoq) * ordering_cost
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("New Weekly Demand", f"{new_demand:,.0f}", delta=f"{demand_change}%")
    col2.metric("New EOQ", f"{new_eoq:,.0f}")
    col3.metric("New Safety Stock", f"{new_ss:,.0f}")
    col4.metric("New Annual Cost", f"${new_cost:,.2f}")
    st.caption("EOQ/Safety Stock assumptions: EOQ=$500 ordering cost, 25% holding. SS at 95% service level (z=1.65).")

with tab4:
    st.header("Stockout Risk Predictor")
    selected_dept4 = st.selectbox("Select Department", class_a_depts, key="dept_stockout")
    inv_row4 = inventory[inventory['Dept'] == selected_dept4].iloc[0]
    current_stock4 = st.number_input("Current Stock Level (units)", min_value=0, value=int(inv_row4['ROP'] * 2), step=100)
    lead_time4 = st.number_input("Lead Time (weeks)", min_value=1, max_value=12, value=2)
    
    avg_demand = inv_row4['Avg_Weekly_Demand']
    demand_std = df[df['Dept'] == selected_dept4].groupby('Date')['Weekly_Sales'].sum().std()
    
    expected_consumption = avg_demand * lead_time4
    consumption_std = demand_std * np.sqrt(lead_time4)
    
    stockout_prob = (1 - stats.norm.cdf((current_stock4 - expected_consumption) / consumption_std)) * 100 if consumption_std > 0 else 0
    
    if stockout_prob < 10: st.success(f"LOW RISK — {stockout_prob:.1f}% prob")
    elif stockout_prob < 30: st.warning(f"MEDIUM RISK — {stockout_prob:.1f}% prob")
    else: st.error(f"HIGH RISK — {stockout_prob:.1f}% prob. Reorder immediately.")
    
    st.caption("Risk: <10% acceptable (95% service level), 10-30% monitor, >30% critical.")
    st.metric("Days of Sales Coverage", f"{current_stock4 / (avg_demand / 7):.0f} days")
    st.caption("Based on average revenue/week.")

with tab5:
    st.header("ABC Inventory Classification")
    fig = px.bar(abc.head(30), x='Dept', y='Annual_Sales_M', color='ABC_Category', title='Annual Sales by Department (Top 30)')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(abc[['Dept', 'Annual_Sales_M', 'Sales_%', 'Cumulative_%', 'ABC_Category']], use_container_width=True)