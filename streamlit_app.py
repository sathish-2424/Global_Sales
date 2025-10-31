import streamlit as st
st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide", page_icon="📊")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import time
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ===== CONFIG =====
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
MODEL_CACHE_DIR = 'models_cache'

# ===== HELPER FUNCTIONS =====
@st.cache_data(show_spinner=False)
def load_data_from_file(uploaded_file):
    """Load CSV with error handling"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            return df, True
        
        if os.path.exists('sales_data.csv'):
            df = pd.read_csv('sales_data.csv', encoding='latin1')
            return df, True
        
        return None, False
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None, False

def basic_cleaning(df):
    """Clean dataset safely"""
    try:
        df = df.copy()
        
        # Drop unnecessary columns
        to_drop = [c for c in ['ADDRESSLINE2', 'ADDRESSLINE3', 'ADDRESSLINE4', 'TERRITORY'] 
                  if c in df.columns]
        df.drop(columns=to_drop, inplace=True, errors='ignore')
        
        # Rename for consistency
        if 'ADDRESSLINE1' in df.columns:
            df.rename(columns={'ADDRESSLINE1': 'ADDRESS'}, inplace=True)
        
        # Drop rows with missing postal code (if exists)
        if 'POSTALCODE' in df.columns:
            df.dropna(subset=['POSTALCODE'], inplace=True)
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error during cleaning: {str(e)}")
        return None

def create_features(df):
    """Create engineered features safely"""
    try:
        df = df.copy()
        
        # ===== Price-based features =====
        if all(col in df.columns for col in ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']):
            # Profit margin calculation (safe division)
            sales_safe = df['SALES'].replace(0, np.nan)
            df['PROFIT_MARGIN'] = (df['SALES'] - df['QUANTITYORDERED'] * (df['PRICEEACH'] * 0.7)) / sales_safe
            
            # Price discount
            msrp_safe = df['MSRP'].replace(0, np.nan)
            df['PRICE_DISCOUNT'] = (df['MSRP'] - df['PRICEEACH']) / msrp_safe
            
            # Revenue per unit
            qty_safe = df['QUANTITYORDERED'].replace(0, np.nan)
            df['REVENUE_PER_UNIT'] = df['SALES'] / qty_safe
        else:
            df['PROFIT_MARGIN'] = 0
            df['PRICE_DISCOUNT'] = 0
            df['REVENUE_PER_UNIT'] = 0
        
        # ===== Temporal features =====
        if 'ORDERDATE' in df.columns:
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
            df['DAY_OF_WEEK'] = df['ORDERDATE'].dt.dayofweek
            df['DAY_OF_MONTH'] = df['ORDERDATE'].dt.day
            df['WEEK_OF_YEAR'] = df['ORDERDATE'].dt.isocalendar().week.astype('Int64')
        else:
            df['DAY_OF_WEEK'] = 0
            df['DAY_OF_MONTH'] = 0
            df['WEEK_OF_YEAR'] = 0
        
        # ===== Customer aggregation =====
        if 'CUSTOMERNAME' in df.columns and len(df) > 0:
            try:
                customer_stats = df.groupby('CUSTOMERNAME').agg({
                    'ORDERNUMBER': 'count',
                    'SALES': ['sum', 'mean'],
                    'QUANTITYORDERED': 'sum'
                }).reset_index()
                customer_stats.columns = ['CUSTOMERNAME', 'CUSTOMER_ORDER_COUNT', 
                                        'CUSTOMER_TOTAL_SALES', 'CUSTOMER_AVG_SALES', 
                                        'CUSTOMER_TOTAL_QTY']
                df = df.merge(customer_stats, on='CUSTOMERNAME', how='left')
            except Exception as e:
                st.warning(f"Could not aggregate customer stats: {e}")
                df['CUSTOMER_ORDER_COUNT'] = 0
                df['CUSTOMER_TOTAL_SALES'] = 0
                df['CUSTOMER_AVG_SALES'] = 0
                df['CUSTOMER_TOTAL_QTY'] = 0
        else:
            df['CUSTOMER_ORDER_COUNT'] = 0
            df['CUSTOMER_TOTAL_SALES'] = 0
            df['CUSTOMER_AVG_SALES'] = 0
            df['CUSTOMER_TOTAL_QTY'] = 0
        
        # ===== Product aggregation =====
        if 'PRODUCTLINE' in df.columns and len(df) > 0:
            try:
                product_stats = df.groupby('PRODUCTLINE').agg({
                    'SALES': ['mean', 'sum'],
                    'QUANTITYORDERED': 'mean'
                }).reset_index()
                product_stats.columns = ['PRODUCTLINE', 'PRODUCTLINE_AVG_SALES', 
                                        'PRODUCTLINE_TOTAL_SALES', 'PRODUCTLINE_AVG_QTY']
                df = df.merge(product_stats, on='PRODUCTLINE', how='left')
            except Exception as e:
                st.warning(f"Could not aggregate product stats: {e}")
                df['PRODUCTLINE_AVG_SALES'] = 0
                df['PRODUCTLINE_TOTAL_SALES'] = 0
                df['PRODUCTLINE_AVG_QTY'] = 0
        else:
            df['PRODUCTLINE_AVG_SALES'] = 0
            df['PRODUCTLINE_TOTAL_SALES'] = 0
            df['PRODUCTLINE_AVG_QTY'] = 0
        
        # ===== Interaction features =====
        df['QTY_PRICE_INTERACTION'] = (df.get('QUANTITYORDERED', 0) * 
                                       df.get('PRICEEACH', 0))
        df['STATUS_MSRP_RATIO'] = ((df.get('STATUS', '') == 'Shipped').astype(int) * 
                                   df.get('MSRP', 0))
        
        # ===== Final NaN fill =====
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error during feature engineering: {str(e)}")
        return None

def build_feature_matrix(df):
    """Build X, y with error handling"""
    try:
        if 'SALES' not in df.columns:
            raise ValueError("Target column 'SALES' not found in dataset")
        
        y = df['SALES'].copy()
        
        # Define features
        categorical_features = [c for c in ['STATUS', 'PRODUCTLINE', 'COUNTRY', 'DEALSIZE'] 
                               if c in df.columns]
        numerical_features = [c for c in [
            'QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'ORDERLINENUMBER', 'QTR_ID', 
            'MONTH_ID', 'YEAR_ID', 'DAY_OF_WEEK', 'DAY_OF_MONTH', 'WEEK_OF_YEAR',
            'PROFIT_MARGIN', 'PRICE_DISCOUNT', 'REVENUE_PER_UNIT',
            'CUSTOMER_ORDER_COUNT', 'CUSTOMER_TOTAL_SALES', 'CUSTOMER_AVG_SALES', 
            'CUSTOMER_TOTAL_QTY', 'PRODUCTLINE_AVG_SALES', 'PRODUCTLINE_TOTAL_SALES', 
            'PRODUCTLINE_AVG_QTY', 'QTY_PRICE_INTERACTION', 'STATUS_MSRP_RATIO'
        ] if c in df.columns]
        
        if not numerical_features:
            raise ValueError("No numerical features found in dataset")
        
        X = df[categorical_features + numerical_features].copy()
        
        # Encode categoricals
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Fill NaNs after selection
        X[numerical_features] = X[numerical_features].fillna(X[numerical_features].median())
        
        return X, y, label_encoders
    
    except Exception as e:
        st.error(f"❌ Error building feature matrix: {str(e)}")
        return None, None, None

def train_and_compare_models(X_train, X_test, y_train, y_test, retrain=False):
    """Train all models with caching"""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        results = {}
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(MODEL_CACHE_DIR, 'scaler.joblib'))
        
        # ===== XGBoost with GridSearch =====
        xgb_path = os.path.join(MODEL_CACHE_DIR, 'xgb_best.joblib')
        best_params = None
        
        if retrain or not os.path.exists(xgb_path):
            with st.spinner("🔄 Training XGBoost (GridSearch)..."):
                xgb_base = XGBRegressor(random_state=42, n_jobs=-1, 
                                       objective='reg:squarederror', verbosity=0)
                param_grid = {
                    'n_estimators': [100, 150],
                    'max_depth': [5, 7],
                    'learning_rate': [0.01, 0.1]
                }
                grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring='r2', 
                                  n_jobs=-1, verbose=0)
                grid.fit(X_train_scaled, y_train)
                best_xgb = grid.best_estimator_
                best_params = grid.best_params_
                joblib.dump(best_xgb, xgb_path)
        else:
            best_xgb = joblib.load(xgb_path)
        
        # ===== Other Models =====
        models_config = [
            ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=10, 
                                                   random_state=42, n_jobs=-1)),
            ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=7, 
                                                           random_state=42)),
            ('Linear Regression', LinearRegression()),
            ('Ridge Regression', Ridge(alpha=1.0))
        ]
        
        for model_name, model in models_config:
            model_path = os.path.join(MODEL_CACHE_DIR, f'{model_name.lower().replace(" ", "_")}.joblib')
            
            if retrain or not os.path.exists(model_path):
                with st.spinner(f"Training {model_name}..."):
                    model.fit(X_train_scaled, y_train)
                    joblib.dump(model, model_path)
            else:
                model = joblib.load(model_path)
            
            # Evaluate
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            results[model_name] = {
                'model': model,
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'y_pred_test': y_pred_test
            }
        
        # Add XGBoost results
        y_pred_train_xgb = best_xgb.predict(X_train_scaled)
        y_pred_test_xgb = best_xgb.predict(X_test_scaled)
        
        results['XGBoost'] = {
            'model': best_xgb,
            'train_r2': r2_score(y_train, y_pred_train_xgb),
            'test_r2': r2_score(y_test, y_pred_test_xgb),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_xgb)),
            'mae': mean_absolute_error(y_test, y_pred_test_xgb),
            'y_pred_test': y_pred_test_xgb
        }
        
        return results, best_params, scaler
    
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        return None, None, None

# ===== SIDEBAR CONTROLS =====
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV (sales_data.csv format)", type=['csv'])
use_default = st.sidebar.checkbox("Use default 'sales_data.csv'", value=True)
retrain = st.sidebar.checkbox("🔄 Retrain models (may be slow)", value=False)

st.sidebar.markdown("---")
nav = st.sidebar.radio("📍 Navigate", 
    ["Data Overview", "EDA", "Feature Engineering", "Modeling", "Results"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Sales Dashboard v2.0 | Fixed & Optimized")

# ===== LOAD DATA =====
with st.spinner("⏳ Loading data..."):
    if uploaded_file is not None:
        df, success = load_data_from_file(uploaded_file)
        if success:
            st.sidebar.success(f"✅ Loaded {uploaded_file.name}")
    elif use_default:
        df, success = load_data_from_file(None)
    else:
        df = None
        success = False

if not success or df is None or df.empty:
    st.error("❌ No data loaded. Upload a CSV or place 'sales_data.csv' in this folder.")
    st.stop()

# ===== MAIN HEADER =====
st.title("💰 Sales Prediction Dashboard")
st.markdown("ML-powered sales forecasting with advanced feature engineering.")

col1, col2, col3 = st.columns(3)
col1.metric("📊 Total Records", f"{len(df):,}")
col2.metric("📈 Columns", df.shape[1])
col3.metric("💾 Memory Usage", f"{df.memory_usage().sum() / 1024**2:.2f} MB")

# ===== PAGE: DATA OVERVIEW =====
if nav == "Data Overview":
    st.header("📋 Dataset Preview & Info")
    
    with st.expander("🔍 Dataset Overview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe().round(3), use_container_width=True)
    
    if 'SALES' in df.columns:
        st.subheader("🏆 Top 10 Orders by Sales")
        top_cols = [c for c in ['ORDERNUMBER', 'CUSTOMERNAME', 'SALES', 'QUANTITYORDERED'] 
                   if c in df.columns]
        st.dataframe(df.sort_values('SALES', ascending=False)[top_cols].head(10), 
                    use_container_width=True)

# ===== PAGE: EDA =====
elif nav == "EDA":
    st.header("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales distribution
        if 'SALES' in df.columns:
            st.subheader("Sales Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['SALES'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_title('Sales Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sales ($)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
    
    with col2:
        # Sales by Product Line
        if 'PRODUCTLINE' in df.columns and 'SALES' in df.columns:
            st.subheader("Total Sales by Product Line")
            product_sales = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            product_sales.plot(kind='barh', ax=ax, color='coral')
            ax.set_title('Total Sales by Product Line', fontsize=12, fontweight='bold')
            st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by Year
        if 'YEAR_ID' in df.columns and 'SALES' in df.columns:
            st.subheader("Average Sales by Year")
            year_sales = df.groupby('YEAR_ID')['SALES'].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(year_sales.index, year_sales.values, marker='o', linewidth=2, markersize=8)
            ax.set_title('Average Sales by Year', fontsize=12, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Sales ($)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with col2:
        # Quantity vs Sales
        if 'QUANTITYORDERED' in df.columns and 'SALES' in df.columns:
            st.subheader("Quantity vs Sales")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df['QUANTITYORDERED'], df['SALES'], alpha=0.5, s=20, color='green')
            ax.set_xlabel('Quantity Ordered')
            ax.set_ylabel('Sales ($)')
            ax.set_title('Quantity vs Sales', fontsize=12, fontweight='bold')
            st.pyplot(fig)
    
    # Sales by Status
    if 'STATUS' in df.columns and 'SALES' in df.columns:
        st.subheader("Total Sales by Order Status")
        status_sales = df.groupby('STATUS')['SALES'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        status_sales.plot(kind='bar', ax=ax, color='mediumpurple')
        ax.set_title('Total Sales by Order Status', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Sales ($)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

# ===== PAGE: FEATURE ENGINEERING =====
elif nav == "Feature Engineering":
    st.header("⚙️ Feature Engineering")
    st.write("Automated feature creation from raw dataset")
    
    with st.spinner("🔄 Engineering features..."):
        df_clean = basic_cleaning(df)
        if df_clean is None:
            st.stop()
        
        df_features = create_features(df_clean)
        if df_features is None:
            st.stop()
    
    st.subheader("Engineered Dataset Preview")
    st.dataframe(df_features.head(10), use_container_width=True)
    
    new_features = [c for c in df_features.columns if c not in df.columns]
    st.subheader(f"✨ New Features Created ({len(new_features)})")
    st.write(new_features)
    
    st.markdown("---")
    csv = df_features.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Engineered Dataset", csv, "sales_engineered.csv", "text/csv")

# ===== PAGE: MODELING =====
elif nav == "Modeling":
    st.header("🤖 Model Training & Comparison")
    
    with st.spinner("⏳ Preparing features..."):
        df_clean = basic_cleaning(df)
        if df_clean is None:
            st.stop()
        
        df_features = create_features(df_clean)
        if df_features is None:
            st.stop()
        
        X, y, le_dict = build_feature_matrix(df_features)
        if X is None:
            st.stop()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Feature Matrix Shape", f"{X.shape[0]} × {X.shape[1]}")
    col2.metric("🎯 Target Shape", f"{y.shape[0]},")
    col3.metric("💡 Features", X.shape[1])
    
    # Train-test split
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100.0, random_state=42
    )
    
    col1, col2 = st.columns(2)
    col1.metric("📚 Train Samples", f"{X_train.shape[0]:,}")
    col2.metric("🧪 Test Samples", f"{X_test.shape[0]:,}")
    
    if retrain:
        st.warning("🔄 Retraining all models. This may take several minutes...")
    else:
        st.info("💾 Using cached models if available. Check 'Retrain models' to force retrain.")
    
    # Train models
    with st.spinner("⏳ Training models..."):
        start = time.time()
        results, best_params, scaler = train_and_compare_models(
            X_train, X_test, y_train, y_test, retrain=retrain
        )
        elapsed = time.time() - start
    
    if results is None:
        st.stop()
    
    st.success(f"✅ Training complete ({elapsed:.1f}s)")
    
    # Best params
    if best_params:
        st.subheader("🏆 XGBoost - Best Hyperparameters (GridSearch)")
        for param, value in best_params.items():
            st.write(f"**{param}:** {value}")
    
    # Model comparison
    st.subheader("📊 Model Comparison")
    comparison_df = pd.DataFrame([{
        'Model': name,
        'Train R²': round(v['train_r2'], 4),
        'Test R²': round(v['test_r2'], 4),
        'RMSE ($)': f"{v['rmse']:,.2f}",
        'MAE ($)': f"{v['mae']:,.2f}"
    } for name, v in results.items()])
    
    st.dataframe(comparison_df.set_index('Model'), use_container_width=True)
    
    st.markdown("---")
    st.info("💾 Models saved to `models_cache/` directory for reuse")

# ===== PAGE: RESULTS =====
elif nav == "Results":
    st.header("📈 Results & Analysis")
    
    # Prepare features and load models
    with st.spinner("⏳ Preparing results..."):
        df_clean = basic_cleaning(df)
        if df_clean is None:
            st.stop()
        
        df_features = create_features(df_clean)
        if df_features is None:
            st.stop()
        
        X, y, le_dict = build_feature_matrix(df_features)
        if X is None:
            st.stop()
    
    # Load models
    scaler_path = os.path.join(MODEL_CACHE_DIR, 'scaler.joblib')
    xgb_path = os.path.join(MODEL_CACHE_DIR, 'xgb_best.joblib')
    
    if not os.path.exists(scaler_path) or not os.path.exists(xgb_path):
        st.error("❌ Trained models not found. Go to 'Modeling' and train models first.")
        st.stop()
    
    try:
        scaler = joblib.load(scaler_path)
        xgb = joblib.load(xgb_path)
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()
    
    # Predictions
    X_scaled = scaler.transform(X)
    y_pred = xgb.predict(X_scaled)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 R² Score", f"{r2:.4f}")
    col2.metric("📉 RMSE", f"${rmse:,.2f}")
    col3.metric("📍 MAE", f"${mae:,.2f}")
    
    st.divider()
    
    # Feature importance
    st.subheader("🎯 Feature Importance (XGBoost)")
    try:
        fi_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': xgb.feature_importances_
        }).sort_values('Importance', ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set_title('Top 20 Important Features', fontsize=12, fontweight='bold')
        st.pyplot(fig)
        
        st.dataframe(fi_df, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not extract feature importance: {e}")
    
    st.divider()
    
    # Predictions vs Actual
    st.subheader("📊 Predictions vs Actual")
    sample_n = st.slider("Number of points to plot", 100, min(5000, len(y)), 1000)
    sample_idx = np.random.choice(len(y), min(sample_n, len(y)), replace=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y.iloc[sample_idx], y_pred[sample_idx], alpha=0.5, s=30, color='steelblue')
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Sales ($)', fontsize=11)
    ax.set_ylabel('Predicted Sales ($)', fontsize=11)
    ax.set_title('Actual vs Predicted (XGBoost)', fontsize=12, fontweight='bold')
    ax.legend()
    st.pyplot(fig)
    
    st.divider()
    
    # Residuals
    st.subheader("📐 Residuals Distribution")
    residuals = y.values - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Residual ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Residual", f"${residuals.mean():,.2f}")
    col2.metric("Std Dev Residual", f"${residuals.std():,.2f}")
    col3.metric("Median Residual", f"${np.median(residuals):,.2f}")
    
    st.divider()
    
    # Top customers analysis
    if 'CUSTOMERNAME' in df_features.columns:
        st.subheader("🏢 Top Customers (Actual vs Predicted)")
        df_compare = df_features.copy()
        df_compare['predicted_sales'] = y_pred
        df_compare['residual'] = residuals
        
        try:
            top_customers = df_compare.groupby('CUSTOMERNAME').agg({
                'SALES': ['sum', 'count'],
                'predicted_sales': 'sum',
                'residual': 'mean'
            }).round(2)
            top_customers.columns = ['Total_Actual', 'Order_Count', 'Total_Predicted', 'Avg_Residual']
            top_customers = top_customers.sort_values('Total_Actual', ascending=False).head(10)
            
            st.dataframe(top_customers, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not analyze customers: {e}")

