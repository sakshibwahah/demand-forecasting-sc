import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
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
    df = df[df['Weekly_Sales'] > 0]
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

features_cols = ['week', 'month', 'quarter', 'year', 'lag_1', 'lag_4',
                 'lag_52', 'rolling_mean_4', 'rolling_mean_12']

df = load_data()
abc, inventory, forecast_metrics = load_outputs()
models = load_models()
class_a_depts = abc[abc['ABC_Category'] == 'A']['Dept'].tolist()

st.title("Supply Chain Analytics Dashboard")
st.markdown("Demand forecasting and inventory optimisation across Walmart's Class A departments")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Demand Forecast",
    "Inventory Recommendations",
    "Scenario Simulator",
    "Stockout Risk",
    "ABC Analysis"
])

# ── TAB 1: DEMAND FORECAST ─────────────────────────────────
with tab1:
    st.header("Demand Forecast by Department")

    selected_dept = st.selectbox("Select Department", class_a_depts, key="dept_forecast")

    dept_df = df[df['Dept'] == selected_dept].groupby('Date')['Weekly_Sales'].mean().reset_index()
    dept_df.columns = ['ds', 'y']
    dept_df = dept_df.sort_values('ds').reset_index(drop=True)

    dept_feat = create_features(dept_df).dropna()
    split_idx = int(len(dept_feat) * 0.8)
    train_d = dept_feat.iloc[:split_idx]
    test_d = dept_feat.iloc[split_idx:]

    model = models[selected_dept]
    preds = model.predict(test_d[features_cols])

    mape_vals = forecast_metrics[forecast_metrics['Dept'] == selected_dept]['MAPE_%'].values
    mape_val = mape_vals[0] if len(mape_vals) > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Department", f"Dept {selected_dept}")
    col2.metric("Forecast MAPE", f"{mape_val:.2f}%")
    col3.metric("ABC Category", "A — High Priority")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_d['ds'], y=train_d['y'],
                             mode='lines', name='Actual (Train)',
                             line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=test_d['ds'], y=test_d['y'],
                             mode='lines', name='Actual (Test)',
                             line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test_d['ds'], y=preds,
                             mode='lines', name='XGBoost Forecast',
                             line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'Weekly Sales Forecast — Dept {selected_dept}',
                      xaxis_title='Date', yaxis_title='Weekly Sales (USD)',
                      height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Demand aggregated as average weekly sales per store. XGBoost model trained on 80% of historical data, evaluated on 20% holdout.")

# ── TAB 2: INVENTORY RECOMMENDATIONS ──────────────────────
with tab2:
    st.header("Inventory Recommendations")
    st.caption("Inventory metrics represent per-store average demand. Ordering cost $2,000 reflects retail PO processing, fixed transportation, and supplier coordination. Holding cost 25% covers capital (~15%), storage (~5%), and shrinkage (~5%) — consistent with Chopra & Meindl (2016).")
    selected_dept2 = st.selectbox("Select Department", class_a_depts, key="dept_inventory")
    inv_row = inventory[inventory['Dept'] == selected_dept2].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Weekly Demand", f"{inv_row['Avg_Weekly_Demand']:,.0f} units")
    col2.metric("EOQ", f"{inv_row['EOQ']:,.0f} units")
    col3.metric("Safety Stock (95%)", f"{inv_row['Safety_Stock_95']:,.0f} units")
    col4.metric("Reorder Point", f"{inv_row['ROP']:,.0f} units")

    st.subheader("Cost Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Policy Cost (2-week cycle)", f"${inv_row['Arbitrary_Annual_Cost']:,.2f}")
    col2.metric("EOQ Optimal Cost", f"${inv_row['EOQ_Annual_Cost']:,.2f}")
    col3.metric("Cost Reduction", f"{inv_row['Cost_Reduction_%']:.2f}%")
    st.success(f"EOQ policy saves {inv_row['Cost_Reduction_%']:.2f}% vs baseline 2-week ordering cycle")
    st.caption("Baseline policy assumes ordering every 2 weeks (common retail replenishment cycle). EOQ minimises total ordering + holding cost.")

    fig = go.Figure(go.Bar(
        x=['Baseline Policy (2-week cycle)', 'EOQ Optimal'],
        y=[inv_row['Arbitrary_Annual_Cost'], inv_row['EOQ_Annual_Cost']],
        marker_color=['#ef553b', '#00cc96'],
        text=[f"${inv_row['Arbitrary_Annual_Cost']:,.0f}", f"${inv_row['EOQ_Annual_Cost']:,.0f}"],
        textposition='auto'
    ))
    fig.update_layout(title=f'Annual Inventory Cost Comparison — Dept {selected_dept2}',
                      yaxis_title='Annual Cost ($)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Replenishment Alert")
    current_stock = st.number_input("Enter current stock level (units)",
                                    min_value=0, value=int(inv_row['ROP']), step=100)
    if current_stock <= inv_row['ROP']:
        st.error(f"REORDER NOW — Current stock ({current_stock:,.0f}) is at or below ROP ({inv_row['ROP']:,.0f})")
    elif current_stock <= inv_row['ROP'] * 1.15:
        st.warning(f"MONITOR — Current stock ({current_stock:,.0f}) is within 15% of ROP ({inv_row['ROP']:,.0f})")
    else:
        st.success(f"SAFE — Current stock ({current_stock:,.0f}) is above ROP ({inv_row['ROP']:,.0f})")
    st.caption("ROP = (Avg weekly demand × lead time) + safety stock at 95% service level. Monitor zone triggers at 15% above ROP to allow time for reorder processing.")

# ── TAB 3: SCENARIO SIMULATOR ──────────────────────────────
with tab3:
    st.header("Scenario Simulator")
    st.markdown("Adjust demand and cost assumptions to see how inventory policy should change in real time.")

    selected_dept3 = st.selectbox("Select Department", class_a_depts, key="dept_scenario")
    inv_row3 = inventory[inventory['Dept'] == selected_dept3].iloc[0]

    demand_change = st.slider("Demand Change (%)", min_value=-50, max_value=100, value=0, step=5)

    st.markdown("#### Cost & Supply Assumptions")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        unit_cost = st.number_input("Unit Cost ($)", min_value=10.0, value=50.0, step=5.0)
    with col_b:
        ordering_cost = st.number_input("Ordering Cost ($)", min_value=1.0, value=2000.0, step=50.0)
    with col_c:
        holding_cost_pct = st.number_input("Annual Holding Cost (%)", min_value=1, max_value=100, value=25)
    with col_d:
        lead_time = st.number_input("Lead Time (weeks)", min_value=1, value=2, step=1, key="lt_tab3")

    holding_cost = unit_cost * (holding_cost_pct / 100)
    base_demand = inv_row3['Avg_Weekly_Demand']
    new_demand = base_demand * (1 + demand_change / 100)
    new_annual = new_demand * 52

    new_eoq = np.sqrt((2 * new_annual * ordering_cost) / holding_cost)
    demand_std3 = df[df['Dept'] == selected_dept3].groupby('Date')['Weekly_Sales'].mean().std()
    new_ss = 1.65 * demand_std3 * np.sqrt(lead_time)
    new_rop = (new_demand * lead_time) + new_ss
    new_cost = (new_eoq / 2) * holding_cost + (new_annual / new_eoq) * ordering_cost
    cost_change = ((new_cost - inv_row3['EOQ_Annual_Cost']) / inv_row3['EOQ_Annual_Cost']) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("New Weekly Demand", f"{new_demand:,.0f}", delta=f"{demand_change}%")
    col2.metric("New EOQ", f"{new_eoq:,.0f}",
                delta=f"{((new_eoq - inv_row3['EOQ']) / inv_row3['EOQ'] * 100):.1f}%")
    col3.metric("New Safety Stock", f"{new_ss:,.0f}", 
            help="Safety stock depends on demand variability (σ), not demand level. It changes only with lead time or service level.")
    col4.metric("New Annual Cost", f"${new_cost:,.2f}", delta=f"{cost_change:.1f}%")

    st.caption(f"EOQ formula: √(2DS/H). Ordering cost ${ordering_cost:.0f}/order, holding cost {holding_cost_pct}% of unit cost annually. Safety stock at 95% service level (z=1.65) based on historical demand variability — independent of demand level changes.")

    demand_range = np.linspace(-50, 100, 100)
    eoq_range = [np.sqrt((2 * base_demand * (1 + d/100) * 52 * ordering_cost) / holding_cost)
                 for d in demand_range]
    cost_range = [(e/2)*holding_cost + (base_demand*(1+d/100)*52/e)*ordering_cost
                  for d, e in zip(demand_range, eoq_range)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=demand_range, y=cost_range, mode='lines',
                             name='Annual Cost', line=dict(color='steelblue', width=2)))
    fig.add_vline(x=demand_change, line_dash='dash', line_color='red',
                  annotation_text=f"Current: {demand_change}%")
    fig.update_layout(title='Annual Inventory Cost vs Demand Change',
                      xaxis_title='Demand Change (%)',
                      yaxis_title='Annual Cost ($)', height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: STOCKOUT RISK ───────────────────────────────────
with tab4:
    st.header("Stockout Risk Predictor")
    st.caption("Stockout probability assumes normally distributed demand during lead time (standard inventory theory). Risk thresholds: <10% acceptable (95% service level target), 10–30% monitor, >30% critical.")

    selected_dept4 = st.selectbox("Select Department", class_a_depts, key="dept_stockout")
    inv_row4 = inventory[inventory['Dept'] == selected_dept4].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        current_stock4 = st.number_input("Current Stock Level (units)",
                                         min_value=0,
                                         value=int(inv_row4['ROP'] * 1.1),
                                         step=100)
    with col2:
        lead_time4 = st.number_input("Lead Time (weeks)", min_value=1, max_value=12, value=2)

    avg_demand = inv_row4['Avg_Weekly_Demand']
    demand_std4 = df[df['Dept'] == selected_dept4].groupby('Date')['Weekly_Sales'].mean().std()

    expected_consumption = avg_demand * lead_time4
    consumption_std = demand_std4 * np.sqrt(lead_time4)

    if consumption_std > 0:
        z_score = (current_stock4 - expected_consumption) / consumption_std
        stockout_prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        stockout_prob = 0

    st.subheader("Stockout Probability During Lead Time")
    if stockout_prob < 10:
        st.success(f"LOW RISK — {stockout_prob:.1f}% probability of stockout")
    elif stockout_prob < 30:
        st.warning(f"MEDIUM RISK — {stockout_prob:.1f}% probability of stockout")
    else:
        st.error(f"HIGH RISK — {stockout_prob:.1f}% probability of stockout. Reorder immediately.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Consumption During Lead Time", f"{expected_consumption:,.0f} units")
    col2.metric("Current Stock", f"{current_stock4:,.0f} units")
    col3.metric("Stockout Probability", f"{stockout_prob:.1f}%")

    days_coverage = current_stock4 / (avg_demand / 7) if avg_demand > 0 else 0
    st.metric("Days of Sales Coverage", f"{days_coverage:.0f} days")
    st.caption("Days of Sales Coverage based on average weekly revenue rate. Dataset reports revenue not physical unit counts.")

# ── TAB 5: ABC ANALYSIS ────────────────────────────────────
with tab5:
    st.header("ABC Inventory Classification")
    st.caption("ABC boundaries of 70%/90% follow standard APICS inventory classification methodology (Pareto principle). Class A departments drive ~70% of revenue and are prioritised for tight inventory control and accurate demand forecasting.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Class A Departments", "21", delta="69.6% of revenue")
    col2.metric("Class B Departments", "17", delta="19.9% of revenue")
    col3.metric("Class C Departments", "43", delta="10.6% of revenue")

    fig = px.bar(abc, x='Dept', y='Annual_Sales_M',
             color='ABC_Category',
             color_discrete_map={'A': 'steelblue', 'B': 'orange', 'C': 'lightgray'},
             title='Annual Sales by Department (All Departments)',
             labels={'Annual_Sales_M': 'Annual Sales (USD M)', 'Dept': 'Department'})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Classification Table")
    st.dataframe(abc[['Dept', 'Annual_Sales_M', 'Sales_%', 'Cumulative_%', 'ABC_Category']]
                 .rename(columns={'Annual_Sales_M': 'Annual Sales (USD M)',
                                  'Sales_%': 'Sales Share (%)',
                                  'Cumulative_%': 'Cumulative (%)',
                                  'ABC_Category': 'Category'}),
                 use_container_width=True)