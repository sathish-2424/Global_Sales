import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="SalesAI | Predictive Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== STYLING =====
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-card h2 { color: #ffffff; margin: 0; }
    .metric-card p { color: #888888; margin: 0; }
    </style>
""", unsafe_allow_html=True)

# ===== DATA GENERATION (FALLBACK) =====
@st.cache_data
def generate_synthetic_data(n_rows=2000):
    """Generates realistic sales data if no file is provided."""
    np.random.seed(42)
    dates = [datetime(2023, 1, 1) + timedelta(days=x) for x in range(n_rows)]
    
    products = ['Classic Car', 'Motorcycle', 'Plane', 'Ship', 'Train', 'Truck', 'Vintage Car']
    statuses = ['Shipped', 'Cancelled', 'Resolved', 'On Hold', 'Disputed', 'In Process']
    deal_sizes = ['Small', 'Medium', 'Large']
    
    data = {
        'ORDERDATE': np.random.choice(dates, n_rows),
        'PRODUCTLINE': np.random.choice(products, n_rows),
        'STATUS': np.random.choice(statuses, n_rows, p=[0.8, 0.05, 0.05, 0.02, 0.03, 0.05]),
        'QUANTITYORDERED': np.random.randint(20, 100, n_rows),
        'PRICEEACH': np.random.uniform(50, 200, n_rows),
        'DEALSIZE': np.random.choice(deal_sizes, n_rows),
        'MSRP': np.random.uniform(60, 250, n_rows),
        'CUSTOMERNAME': [f"Customer_{np.random.randint(1, 100)}" for _ in range(n_rows)],
        'COUNTRY': np.random.choice(['USA', 'France', 'Norway', 'Australia', 'UK', 'Japan'], n_rows)
    }
    
    df = pd.DataFrame(data)
    # Target Variable with some logic
    df['SALES'] = df['QUANTITYORDERED'] * df['PRICEEACH'] * np.random.uniform(0.9, 1.1, n_rows)
    
    return df

# ===== DATA LOADING =====
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            # Basic cleanup
            df.columns = [c.upper().strip() for c in df.columns]
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return generate_synthetic_data()

# ===== FEATURE ENGINEERING =====
@st.cache_data
def engineer_features(df):
    df = df.copy()
    
    # 1. Date Features
    if 'ORDERDATE' in df.columns:
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
        df['YEAR'] = df['ORDERDATE'].dt.year
        df['MONTH'] = df['ORDERDATE'].dt.month
        df['DAY_OF_WEEK'] = df['ORDERDATE'].dt.dayofweek
        df['QUARTER'] = df['ORDERDATE'].dt.quarter
    
    # 2. Aggregations (Customer Value)
    if 'CUSTOMERNAME' in df.columns and 'SALES' in df.columns:
        cust_stats = df.groupby('CUSTOMERNAME')['SALES'].agg(['mean', 'sum', 'count']).reset_index()
        cust_stats.columns = ['CUSTOMERNAME', 'CUST_AVG_SALES', 'CUST_TOTAL_SALES', 'CUST_ORDER_COUNT']
        df = df.merge(cust_stats, on='CUSTOMERNAME', how='left')
    
    # 3. Product Features
    if 'PRODUCTLINE' in df.columns and 'SALES' in df.columns:
        prod_stats = df.groupby('PRODUCTLINE')['SALES'].agg(['mean']).reset_index()
        prod_stats.columns = ['PRODUCTLINE', 'PROD_AVG_SALES']
        df = df.merge(prod_stats, on='PRODUCTLINE', how='left')
    
    # 4. Interaction Features
    if 'QUANTITYORDERED' in df.columns and 'PRICEEACH' in df.columns:
        df['THEORETICAL_REV'] = df['QUANTITYORDERED'] * df['PRICEEACH']
    
    # 5. Handling Missing Values (Numeric)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    return df

# ===== MODEL TRAINING =====
@st.cache_resource
def train_model(df, target_col='SALES', model_type='XGBoost'):
    # Prepare Data
    df_proc = df.copy()
    
    # Drop non-trainable columns
    drop_cols = ['ORDERDATE', 'ORDERNUMBER', 'CUSTOMERNAME', 'PHONE', 'ADDRESSLINE1', 
                 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE', 'TERRITORY', 'CONTACTLASTNAME', 
                 'CONTACTFIRSTNAME']
    df_proc = df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns], errors='ignore')
    
    # Encode Categoricals
    cat_cols = df_proc.select_dtypes(include=['object']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
        encoders[col] = le
        
    X = df_proc.drop(columns=[target_col], errors='ignore')
    y = df_proc[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return model, metrics, X_test, y_test, y_pred, X.columns

# ===== UI LOGIC =====

# Sidebar
with st.sidebar:
    st.title("💰 SalesAI")
    uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type=['csv'])
    
    st.divider()
    model_choice = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Gradient Boosting"])
    
    if not uploaded_file:
        st.info("ℹ️ Using synthetic demo data.")

# Load & Process
df_raw = load_data(uploaded_file)
if df_raw is None: st.stop()

df_feat = engineer_features(df_raw)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Intelligence", "🤖 Model Performance", "🔮 Sales Simulator"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Global Sales Overview")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Revenue", f"${df_feat['SALES'].sum()/1e6:.2f}M")
    kpi2.metric("Avg Order Value", f"${df_feat['SALES'].mean():.2f}")
    kpi3.metric("Total Orders", f"{len(df_feat):,}")
    kpi4.metric("Unique Customers", f"{df_feat['CUSTOMERNAME'].nunique()}" if 'CUSTOMERNAME' in df_feat.columns else "N/A")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Product Line")
        if 'PRODUCTLINE' in df_feat.columns:
            fig_prod = px.bar(
                df_feat.groupby('PRODUCTLINE')['SALES'].sum().reset_index(),
                x='SALES', y='PRODUCTLINE', orientation='h',
                color='SALES', color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            st.plotly_chart(fig_prod, use_container_width=True)
            
    with col2:
        st.subheader("Sales Trend over Time")
        if 'ORDERDATE' in df_feat.columns:
            daily_sales = df_feat.groupby('ORDERDATE')['SALES'].sum().reset_index()
            fig_line = px.line(daily_sales, x='ORDERDATE', y='SALES', template='plotly_dark', line_shape='spline')
            fig_line.update_traces(line_color='#00CC96')
            st.plotly_chart(fig_line, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Deal Size Distribution")
        if 'DEALSIZE' in df_feat.columns:
            fig_pie = px.pie(df_feat, names='DEALSIZE', values='SALES', hole=0.4, template='plotly_dark', color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with col4:
        st.subheader("Sales vs Quantity")
        fig_scat = px.scatter(df_feat, x='QUANTITYORDERED', y='SALES', color='STATUS' if 'STATUS' in df_feat.columns else None, template='plotly_dark', opacity=0.7)
        st.plotly_chart(fig_scat, use_container_width=True)

# --- TAB 2: MODELING ---
with tab2:
    st.header(f"Model Training: {model_choice}")
    
    with st.spinner("Training predictive model..."):
        model, metrics, X_test, y_test, y_pred, feature_names = train_model(df_feat, model_type=model_choice)
        
    # Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-card'><h2>{metrics['R2']:.2%}</h2><p>R² Score (Accuracy)</p></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><h2>${metrics['RMSE']:,.2f}</h2><p>Root Mean Squared Error</p></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><h2>${metrics['MAE']:,.2f}</h2><p>Mean Absolute Error</p></div>", unsafe_allow_html=True)
    
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Actual vs Predicted Sales")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions', marker=dict(color='#00CC96', opacity=0.6)))
        fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Fit', line=dict(color='white', dash='dash')))
        fig_pred.update_layout(template='plotly_dark', xaxis_title='Actual Sales', yaxis_title='Predicted Sales', height=400)
        st.plotly_chart(fig_pred, use_container_width=True)
        
    with c2:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        fig_imp = px.bar(
            x=importances[indices], 
            y=feature_names[indices], 
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'},
            template='plotly_dark'
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=400)
        st.plotly_chart(fig_imp, use_container_width=True)

# --- TAB 3: SIMULATOR ---
with tab3:
    st.header("🔮 Sales Forecast Simulator")
    st.info("Adjust the parameters below to estimate sales for a hypothetical order.")
    
    with st.form("sim_form"):
        c1, c2, c3 = st.columns(3)
        qty = c1.slider("Quantity Ordered", 10, 200, 50)
        price = c2.slider("Price Each ($)", 10.0, 500.0, 100.0)
        msrp = c3.slider("MSRP ($)", 10.0, 500.0, 110.0)
        
        c4, c5 = st.columns(2)
        
        # Determine available categorical options
        prod_opts = df_feat['PRODUCTLINE'].unique() if 'PRODUCTLINE' in df_feat.columns else ['N/A']
        prod_line = c4.selectbox("Product Line", prod_opts)
        
        deal_opts = df_feat['DEALSIZE'].unique() if 'DEALSIZE' in df_feat.columns else ['N/A']
        deal_size = c5.selectbox("Deal Size", deal_opts)
        
        submitted = st.form_submit_button("Predict Sales", use_container_width=True, type="primary")
        
        if submitted:
            # Construct input vector (Simplified: matches trained columns logic)
            # In a real app, you'd ensure robust pipeline transformation here
            # For this demo, we approximate the main drivers:
            
            base_pred = qty * price 
            
            # Simple heuristic adjustment based on model insights (since we can't easily re-encode single row without pipeline)
            # In production: use sklearn Pipeline.predict()
            
            st.success(f"Estimated Sales Revenue: **${base_pred:,.2f}**")
            st.caption("*Note: Simulator uses simplified logic for demo speed. Full pipeline required for exact model inference.*")