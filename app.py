import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SC Analytics Dashboard",
    page_icon="📦",
    layout="wide"
)

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
    
    dept_feat = create_features(dept_df)
    dept_feat = dept_feat.dropna()
    
    split_idx = int(len(dept_feat) * 0.8)
    train_d = dept_feat.iloc[:split_idx]
    test_d = dept_feat.iloc[split_idx:]
    
    model = models[selected_dept]
    preds = model.predict(test_d[features_cols])
    
    mape = forecast_metrics[forecast_metrics['Dept'] == selected_dept]['MAPE_%'].values
    mape_val = mape[0] if len(mape) > 0 else 0
    
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

# ── TAB 2: INVENTORY RECOMMENDATIONS ──────────────────────
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
    col3.metric("Cost Reduction", f"{inv_row['Cost_Reduction_%']:.2f}%", 
            delta=f"-{inv_row['Cost_Reduction_%']:.2f}% cost saved",
            delta_color="inverse")
    
    fig = go.Figure(go.Bar(
        x=['Arbitrary Policy', 'EOQ Optimal'],
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
    elif current_stock <= inv_row['ROP'] * 1.2:
        st.warning(f"MONITOR — Current stock ({current_stock:,.0f}) is within 20% of ROP ({inv_row['ROP']:,.0f})")
    else:
        st.success(f"SAFE — Current stock ({current_stock:,.0f}) is above ROP ({inv_row['ROP']:,.0f})")

# ── TAB 3: SCENARIO SIMULATOR ──────────────────────────────
with tab3:
    st.header("Scenario Simulator")
    st.markdown("Adjust demand to see how inventory policy should change in real time")
    
    selected_dept3 = st.selectbox("Select Department", class_a_depts, key="dept_scenario")
    inv_row3 = inventory[inventory['Dept'] == selected_dept3].iloc[0]
    
    demand_change = st.slider("Demand Change (%)", min_value=-50, max_value=100, value=0, step=5)
    
    ordering_cost = 500
    unit_cost = 25
    holding_cost = unit_cost * 0.25
    lead_time = 2
    z = 1.65
    
    base_demand = inv_row3['Avg_Weekly_Demand']
    new_demand = base_demand * (1 + demand_change / 100)
    new_annual = new_demand * 52
    
    new_eoq = np.sqrt((2 * new_annual * ordering_cost) / holding_cost)
    dept_data = df[df['Dept'] == selected_dept3].groupby('Date')['Weekly_Sales'].mean().reset_index()
    demand_std3 = dept_data['Weekly_Sales'].std()
    new_ss = z * demand_std3 * np.sqrt(lead_time)
    new_rop = (new_demand * lead_time) + new_ss
    new_cost = (new_eoq / 2) * holding_cost + (new_annual / new_eoq) * ordering_cost
    base_cost = inv_row3['EOQ_Annual_Cost']
    cost_change = ((new_cost - base_cost) / base_cost) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("New Weekly Demand", f"{new_demand:,.0f}", delta=f"{demand_change}%")
    col2.metric("New EOQ", f"{new_eoq:,.0f}", 
                delta=f"{((new_eoq - inv_row3['EOQ']) / inv_row3['EOQ'] * 100):.1f}%")
    col3.metric("New Safety Stock", f"{new_ss:,.0f}",
                delta=f"{((new_ss - inv_row3['Safety_Stock_95']) / inv_row3['Safety_Stock_95'] * 100):.1f}%")
    col4.metric("New Annual Cost", f"${new_cost:,.2f}", delta=f"{cost_change:.1f}%")
    
    demand_range = np.linspace(-50, 100, 100)
    eoq_range = [np.sqrt((2 * base_demand * (1 + d/100) * 52 * ordering_cost) / holding_cost) 
                 for d in demand_range]
    cost_range = [(eoq/2)*holding_cost + (base_demand*(1+d/100)*52/eoq)*ordering_cost 
                  for d, eoq in zip(demand_range, eoq_range)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=demand_range, y=cost_range,
                             mode='lines', name='Annual Cost',
                             line=dict(color='steelblue', width=2)))
    fig.add_vline(x=demand_change, line_dash='dash', line_color='red',
                  annotation_text=f"Current: {demand_change}%")
    fig.update_layout(title='Annual Inventory Cost vs Demand Change',
                      xaxis_title='Demand Change (%)',
                      yaxis_title='Annual Cost ($)', height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: STOCKOUT RISK ───────────────────────────────────
with tab4:
    st.header("Stockout Risk Predictor")
    
    selected_dept4 = st.selectbox("Select Department", class_a_depts, key="dept_stockout")
    inv_row4 = inventory[inventory['Dept'] == selected_dept4].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        current_stock4 = st.number_input("Current Stock Level (units)", 
                                  min_value=0, 
                                  value=int(inv_row4['ROP'] * 2),  # change from 1.5 to 2
                                  step=100)
    with col2:
        lead_time4 = st.number_input("Lead Time (weeks)", min_value=1, max_value=12, value=2)
    
    avg_demand = inv_row4['Avg_Weekly_Demand']
    demand_std = avg_demand * 0.25
    
    expected_consumption = avg_demand * lead_time4
    consumption_std = demand_std * np.sqrt(lead_time4)
    
    if consumption_std > 0:
        z_score = (current_stock4 - expected_consumption) / consumption_std
        from scipy import stats
        stockout_prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        stockout_prob = 0
    
    st.subheader("Stockout Probability During Lead Time")
    
    if stockout_prob < 5:
        st.success(f"LOW RISK — {stockout_prob:.1f}% probability of stockout")
    elif stockout_prob < 20:
        st.warning(f"MEDIUM RISK — {stockout_prob:.1f}% probability of stockout")
    else:
        st.error(f"HIGH RISK — {stockout_prob:.1f}% probability of stockout. Reorder immediately.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Consumption", f"{expected_consumption:,.0f} units")
    col2.metric("Current Stock", f"{current_stock4:,.0f} units")
    col3.metric("Stockout Probability", f"{stockout_prob:.1f}%")
    
    days_of_stock = current_stock4 / (avg_demand / 7) if avg_demand > 0 else 0
    st.metric("Days of Stock Remaining", f"{days_of_stock:.0f} days")

# ── TAB 5: ABC ANALYSIS ────────────────────────────────────
with tab5:
    st.header("ABC Inventory Classification")
    
    summary = abc.groupby('ABC_Category').agg(
        Departments=('Dept', 'count'),
        Total_Sales_M=('Annual_Sales_M', 'sum'),
        Sales_Share=('Sales_%', 'sum')
    ).reset_index()
    summary['Total_Sales_M'] = summary['Total_Sales_M'].round(2)
    summary['Sales_Share'] = summary['Sales_Share'].round(2)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Class A Departments", "21", delta="69.6% of revenue")
    col2.metric("Class B Departments", "17", delta="19.9% of revenue")
    col3.metric("Class C Departments", "43", delta="10.6% of revenue")
    
    fig = px.bar(abc.head(30), x='Dept', y='Annual_Sales_M',
                 color='ABC_Category',
                 color_discrete_map={'A': 'steelblue', 'B': 'orange', 'C': 'lightgray'},
                 title='Annual Sales by Department (Top 30)',
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