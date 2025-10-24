import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Advanced Sales Analysis & Prediction Dashboard",
        "Get Help": "https://docs.streamlit.io"
    }
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.main-header {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    text-align: center;
}
.subheader-text {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 20px 0 15px 0;
    border-bottom: 3px solid #667eea;
    padding-bottom: 10px;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 25px;
    border-radius: 12px;
    color: white;
    text-align: center;
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}
.metric-value { font-size: 2rem; font-weight: 700; margin: 10px 0; }
.metric-label { font-size: 0.9rem; opacity: 0.9; font-weight: 500; }
.success-box {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border-left: 5px solid #0d7961;
    padding: 20px;
    border-radius: 8px;
    color: white;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.warning-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-left: 5px solid #d4376b;
    padding: 20px;
    border-radius: 8px;
    color: white;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.info-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    border-left: 5px solid #0080d6;
    padding: 20px;
    border-radius: 8px;
    color: white;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.section-divider {
    margin: 40px 0;
    padding: 20px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab-list"] { gap: 20px; }
.stButton button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    color: white !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# ==================== HELPER FUNCTIONS ====================
@st.cache_data
def load_and_clean_data(file):
    df = pd.read_csv(file, encoding='latin1')
    drop_cols = ['ADDRESSLINE2', 'ADDRESSLINE3', 'ADDRESSLINE4', 'TERRITORY']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.rename(columns={'ADDRESSLINE1': 'ADDRESS'}, inplace=True)
    df.dropna(subset=['POSTALCODE'], inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

@st.cache_data
def create_features(df):
    df_features = df.copy()
    df_features['PROFIT_MARGIN'] = ((df_features['SALES'] - df_features['QUANTITYORDERED'] * 
                                    (df_features['PRICEEACH'] * 0.7)) / df_features['SALES']).fillna(0)
    df_features['PRICE_DISCOUNT'] = ((df_features['MSRP'] - df_features['PRICEEACH']) / df_features['MSRP']).fillna(0)
    df_features['REVENUE_PER_UNIT'] = (df_features['SALES'] / df_features['QUANTITYORDERED']).fillna(0)
    df_features['ORDERDATE'] = pd.to_datetime(df_features['ORDERDATE'])
    df_features['DAY_OF_WEEK'] = df_features['ORDERDATE'].dt.dayofweek
    df_features['DAY_OF_MONTH'] = df_features['ORDERDATE'].dt.day
    df_features['WEEK_OF_YEAR'] = df_features['ORDERDATE'].dt.isocalendar().week
    customer_stats = df_features.groupby('CUSTOMERNAME').agg({
        'ORDERNUMBER': 'count',
        'SALES': ['sum', 'mean']
    }).reset_index()
    customer_stats.columns = ['CUSTOMERNAME', 'CUSTOMER_ORDERS', 'CUSTOMER_TOTAL_SALES', 'CUSTOMER_AVG_SALES']
    df_features = df_features.merge(customer_stats, on='CUSTOMERNAME', how='left')
    return df_features

@st.cache_data
def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=7, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=150, max_depth=7, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'model': model,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'predictions': y_pred
        }
    return results

# ==================== MAIN APP ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<p class="main-header">📊 Sales Prediction Hub</p>', unsafe_allow_html=True)
st.markdown("""<div style="text-align: center; color: #666; margin-bottom: 30px;">
<p style="font-size: 1.1rem; font-weight: 500;">
Advanced Analytics & Machine Learning for Sales Forecasting
</p></div>""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    uploaded_file = st.file_uploader("📂 Upload Sales Dataset", type=["csv"])
    if uploaded_file:
        st.session_state.data_loaded = True
        st.markdown('<div class="success-box">✅ Dataset Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">ℹ️ Please upload a dataset to begin</div>', unsafe_allow_html=True)
        st.stop()

# ==================== MAIN CONTENT ====================
if st.session_state.data_loaded:
    df = load_and_clean_data(uploaded_file)
    df_features = create_features(df)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🔍 Analysis", "⚙️ Feature Engineering", "🤖 Modeling", "📊 Results"])
    
    # ---------- TAB 1: OVERVIEW ----------
    with tab1:
        st.markdown('<p class="subheader-text">Dataset Overview</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        metrics = [("Total Records", df.shape[0]), ("Features", df.shape[1]),
                   ("Total Sales", f"${df['SALES'].sum():,.0f}"), ("Avg Order Value", f"${df['SALES'].mean():,.0f}")]
        for col, (label, value) in zip(cols, metrics):
            col.markdown(f"""<div class="metric-card">
                              <div class="metric-label">📊 {label}</div>
                              <div class="metric-value">{value}</div></div>""", unsafe_allow_html=True)
        st.markdown('<p class="subheader-text">Data Preview</p>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    
    # ---------- TAB 2: ANALYSIS ----------
    with tab2:
        st.markdown('<p class="subheader-text">Exploratory Data Analysis</p>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Histogram(x=df['SALES'], nbinsx=50)])
        fig.update_layout(title="Sales Distribution", xaxis_title="Sales ($)", yaxis_title="Frequency", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    # ---------- TAB 3: FEATURE ENGINEERING ----------
    with tab3:
        st.markdown('<p class="subheader-text">Engineered Features</p>', unsafe_allow_html=True)
        st.dataframe(df_features.head(5), use_container_width=True)
    
    # ---------- TAB 4: MODELING ----------
    with tab4:
        st.markdown('<p class="subheader-text">Model Training & Evaluation</p>', unsafe_allow_html=True)
        y = df_features['SALES']
        categorical_features = ['STATUS', 'PRODUCTLINE', 'COUNTRY', 'DEALSIZE']
        numerical_features = ['QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'ORDERLINENUMBER',
                            'QTR_ID', 'MONTH_ID', 'YEAR_ID', 'DAY_OF_WEEK', 'DAY_OF_MONTH',
                            'WEEK_OF_YEAR', 'PROFIT_MARGIN', 'PRICE_DISCOUNT', 'REVENUE_PER_UNIT',
                            'CUSTOMER_ORDERS', 'CUSTOMER_TOTAL_SALES', 'CUSTOMER_AVG_SALES']
        X = df_features[categorical_features + numerical_features].copy()
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        X[numerical_features] = X[numerical_features].fillna(X[numerical_features].median())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if st.button("🚀 Train All Models"):
            with st.spinner("Training models..."):
                results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
                st.session_state.models_trained = True
                st.session_state.results = results
                st.session_state.X = X
                st.session_state.y_test = y_test
                st.session_state.X_test_scaled = X_test_scaled
            st.markdown('<div class="success-box">✅ Models trained successfully!</div>', unsafe_allow_html=True)
    
    # ---------- TAB 5: RESULTS ----------
    with tab5:
        st.markdown('<p class="subheader-text">Model Performance Comparison</p>', unsafe_allow_html=True)
        if st.session_state.models_trained:
            results = st.session_state.results
            perf_df = pd.DataFrame({
                'Model': list(results.keys()),
                'R² Score': [results[m]['r2'] for m in results.keys()],
                'RMSE ($)': [results[m]['rmse'] for m in results.keys()],
                'MAE ($)': [results[m]['mae'] for m in results.keys()],
                'MAPE (%)': [results[m]['mape']*100 for m in results.keys()]
            })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            best_xgb_model = results['XGBoost']['model']
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_xgb_model.feature_importances_}).sort_values('Importance', ascending=False).head(15)
            fig = go.Figure(data=[go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h')])
            fig.update_layout(title="Top 15 Feature Importances", template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)
            predictions = results['XGBoost']['predictions']
            hover_text = [f"Actual: ${a:,.2f}<br>Predicted: ${p:,.2f}" for a, p in zip(y_test, predictions)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', marker=dict(size=8, color=np.abs(y_test - predictions), colorscale='Viridis', showscale=True, colorbar=dict(title="Error ($)")), text=hover_text, hovertemplate='%{text}<extra></extra>'))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', line=dict(color='red', dash='dash')))
            fig.update_layout(title="Actual vs Predicted Sales", xaxis_title="Actual Sales ($)", yaxis_title="Predicted Sales ($)", template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="warning-box">⚠️ Please train models first!</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""<div style="text-align: center; color: #888; padding: 20px;">
<p> Sales Prediction Dashboard</p>
<p style="font-size: 0.9rem;"></p></div>""", unsafe_allow_html=True)
