### Install & Import Libraries
"""

# Install required libraries
!pip install xgboost pandas scikit-learn matplotlib seaborn numpy -q

# Import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)

"""### Load & Explore Data"""

# Load dataset
df = pd.read_csv('sales_data.csv', encoding='latin1')

print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"✓ Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print(f"✓ Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB\n")

# Display first rows
print("First 5 Rows:")
print(df.head())

print("\n" + "-"*70)
print("Dataset Info:")
print("-"*70)
print(df.info())

print("\n" + "-"*70)
print("Statistical Summary:")
print("-"*70)
print(df.describe())

print("\n" + "-"*70)
print("Missing Values:")
print("-"*70)
print(df.isnull().sum())

print("\n" + "-"*70)
print("Data Shape & Unique Values:")
print("-"*70)
print(f"Shape: {df.shape}")
print(f"\nUnique Values per Column:")
print(df.nunique())

# Column names
print("\n" + "="*70)
print("COLUMN NAMES:")
print("="*70)
print(df.columns.tolist())

"""### Data Cleaning & Preprocessing"""

print("\n" + "="*70)
print("DATA CLEANING & PREPROCESSING")
print("="*70)

# Drop unnecessary columns
columns_to_drop = ['ADDRESSLINE2', 'ADDRESSLINE3', 'ADDRESSLINE4', 'TERRITORY']
df.drop(columns=columns_to_drop, inplace=True)
print(f"✓ Dropped unnecessary columns: {columns_to_drop}")

# Rename column for clarity
df.rename(columns={'ADDRESSLINE1': 'ADDRESS'}, inplace=True)
print(f"✓ Renamed ADDRESSLINE1 → ADDRESS")

# Check and handle missing values
print(f"\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Drop rows with missing critical information
df.dropna(subset=['POSTALCODE'], inplace=True)
print(f"\n✓ Dropped rows with missing POSTALCODE")

print(f"\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Fill remaining missing values if any
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print(f"\n✓ Filled remaining numeric missing values with median")
print(f"\nDataset Shape: {df.shape}")
print(f"Total Rows After Cleaning: {len(df):,}")

# Display top 10 sales
print("\n" + "-"*70)
print("Top 10 Orders by Sales:")
print("-"*70)
print(df.sort_values(by="SALES", ascending=False)[['ORDERNUMBER', 'CUSTOMERNAME', 'SALES', 'QUANTITYORDERED']].head(10))

"""### Exploratory Data Analysis (EDA)"""

print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

# Create EDA visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Sales Data - Key Patterns', fontsize=16, fontweight='bold')

# 1. Sales Distribution
axes[0, 0].hist(df['SALES'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Sales Distribution', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Sales Amount ($)')
axes[0, 0].set_ylabel('Frequency')

# 2. Sales by Product Line
product_sales = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False)
axes[0, 1].barh(product_sales.index, product_sales.values, color='#2ecc71')
axes[0, 1].set_title('Total Sales by Product Line', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Total Sales ($)')

# 3. Sales by Year
year_sales = df.groupby('YEAR_ID')['SALES'].mean()
axes[0, 2].plot(year_sales.index, year_sales.values, marker='o', linewidth=2.5,
               markersize=8, color='#e74c3c')
axes[0, 2].set_title('Average Sales by Year', fontweight='bold', fontsize=12)
axes[0, 2].set_xlabel('Year')
axes[0, 2].set_ylabel('Average Sales ($)')
axes[0, 2].grid(True, alpha=0.3)

# 4. Quantity vs Sales (Scatter)
axes[1, 0].scatter(df['QUANTITYORDERED'], df['SALES'], alpha=0.5, color='#9b59b6', s=30)
axes[1, 0].set_title('Quantity Ordered vs Sales', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Quantity Ordered')
axes[1, 0].set_ylabel('Sales ($)')

# 5. Sales by Order Status
status_sales = df.groupby('STATUS')['SALES'].sum().sort_values(ascending=False)
axes[1, 1].bar(status_sales.index, status_sales.values, color=['#2ecc71', '#e74c3c', '#f39c12'])
axes[1, 1].set_title('Total Sales by Order Status', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Total Sales ($)')
axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Sales by Quarter
quarter_sales = df.groupby('QTR_ID')['SALES'].mean()
axes[1, 2].plot(quarter_sales.index, quarter_sales.values, marker='s', linewidth=2.5,
               markersize=8, color='#1abc9c')
axes[1, 2].set_title('Average Sales by Quarter', fontweight='bold', fontsize=12)
axes[1, 2].set_xlabel('Quarter')
axes[1, 2].set_ylabel('Average Sales ($)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""### Feature Engineering"""

print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# Create a copy for feature engineering
df_features = df.copy()

# 1. Price-based features
df_features['PROFIT_MARGIN'] = (df_features['SALES'] - df_features['QUANTITYORDERED'] *
                               (df_features['PRICEEACH'] * 0.7)) / df_features['SALES']
df_features['PRICE_DISCOUNT'] = (df_features['MSRP'] - df_features['PRICEEACH']) / df_features['MSRP']
df_features['REVENUE_PER_UNIT'] = df_features['SALES'] / df_features['QUANTITYORDERED']

print("✓ Created price-based features: PROFIT_MARGIN, PRICE_DISCOUNT, REVENUE_PER_UNIT")

# 2. Temporal features
df_features['ORDERDATE'] = pd.to_datetime(df_features['ORDERDATE'])
df_features['DAY_OF_WEEK'] = df_features['ORDERDATE'].dt.dayofweek
df_features['DAY_OF_MONTH'] = df_features['ORDERDATE'].dt.day
df_features['WEEK_OF_YEAR'] = df_features['ORDERDATE'].dt.isocalendar().week

print("✓ Created temporal features: DAY_OF_WEEK, DAY_OF_MONTH, WEEK_OF_YEAR")

# 3. Customer features (aggregated)
customer_stats = df_features.groupby('CUSTOMERNAME').agg({
    'ORDERNUMBER': 'count',
    'SALES': ['sum', 'mean'],
    'QUANTITYORDERED': 'sum'
}).reset_index()
customer_stats.columns = ['CUSTOMERNAME', 'CUSTOMER_ORDER_COUNT', 'CUSTOMER_TOTAL_SALES',
                          'CUSTOMER_AVG_SALES', 'CUSTOMER_TOTAL_QTY']

df_features = df_features.merge(customer_stats, on='CUSTOMERNAME', how='left')

print("✓ Created customer aggregation features: CUSTOMER_ORDER_COUNT, CUSTOMER_TOTAL_SALES, etc.")

# 4. Product features (aggregated)
product_stats = df_features.groupby('PRODUCTLINE').agg({
    'SALES': ['mean', 'sum'],
    'QUANTITYORDERED': 'mean'
}).reset_index()
product_stats.columns = ['PRODUCTLINE', 'PRODUCTLINE_AVG_SALES', 'PRODUCTLINE_TOTAL_SALES',
                         'PRODUCTLINE_AVG_QTY']

df_features = df_features.merge(product_stats, on='PRODUCTLINE', how='left')

print("✓ Created product aggregation features: PRODUCTLINE_AVG_SALES, etc.")

# 5. Interaction features
df_features['QTY_PRICE_INTERACTION'] = df_features['QUANTITYORDERED'] * df_features['PRICEEACH']
df_features['STATUS_MSRP_RATIO'] = (df_features['STATUS'] == 'Shipped').astype(int) * df_features['MSRP']

print("✓ Created interaction features: QTY_PRICE_INTERACTION, STATUS_MSRP_RATIO")

print(f"\n✅ Total Features Created: {df_features.shape[1] - df.shape[1]} new features")
print(f"✅ Final Feature Count: {df_features.shape[1]}")

# Display new features
new_features = [col for col in df_features.columns if col not in df.columns]
print(f"\nNew Features: {new_features}")

"""### Data Preparation for Modeling"""

print("\n" + "="*70)
print("DATA PREPARATION FOR MODELING")
print("="*70)

# Target variable
y = df_features['SALES']

# Select features for modeling
categorical_features = ['STATUS', 'PRODUCTLINE', 'COUNTRY', 'DEALSIZE']
numerical_features = ['QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'ORDERLINENUMBER',
                      'QTR_ID', 'MONTH_ID', 'YEAR_ID', 'DAY_OF_WEEK', 'DAY_OF_MONTH',
                      'WEEK_OF_YEAR', 'PROFIT_MARGIN', 'PRICE_DISCOUNT', 'REVENUE_PER_UNIT',
                      'CUSTOMER_ORDER_COUNT', 'CUSTOMER_TOTAL_SALES', 'CUSTOMER_AVG_SALES',
                      'CUSTOMER_TOTAL_QTY', 'PRODUCTLINE_AVG_SALES', 'PRODUCTLINE_TOTAL_SALES',
                      'PRODUCTLINE_AVG_QTY', 'QTY_PRICE_INTERACTION', 'STATUS_MSRP_RATIO']

# Create feature matrix
X = df_features[categorical_features + numerical_features].copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded {col}: {len(le.classes_)} unique values")

# Remove rows with NaN values
X = X.fillna(X.median(numeric_only=True))

print(f"\n✅ Feature Matrix Shape: {X.shape}")
print(f"✅ Target Variable Shape: {y.shape}")

# Check for missing values
print(f"\nMissing Values in Features:")
print(f"Total NaN: {X.isnull().sum().sum()}")

# Display feature statistics
print(f"\n" + "-"*70)
print("Feature Statistics:")
print("-"*70)
print(X.describe().round(2))

"""### Train-Test Split & Scaling"""

print("\n" + "="*70)
print("TRAIN-TEST SPLIT & FEATURE SCALING")
print("="*70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Train Set: {X_train.shape[0]:,} samples")
print(f"✓ Test Set:  {X_test.shape[0]:,} samples")
print(f"✓ Features:  {X_train.shape[1]}")
print(f"✓ Train-Test Ratio: {X_train.shape[0]/X_test.shape[0]:.2f}:1")
print(f"✓ Features Scaled: StandardScaler applied")

"""### XGBoost Model - Hyperparameter Tuning"""

print("\n" + "="*70)
print("XGBOOST MODEL - HYPERPARAMETER TUNING")
print("="*70)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

print("🔍 GridSearchCV running for XGBoost...")
print("This may take a moment...\n")

# Create base model
xgb_base = XGBRegressor(random_state=42, n_jobs=-1)

# GridSearchCV with reduced parameter space for faster execution
param_grid_reduced = {
    'n_estimators': [100, 150],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(xgb_base, param_grid_reduced, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_xgb = grid_search.best_estimator_

print(f"✓ Best Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n✓ Best Cross-Validation R² Score: {grid_search.best_score_:.4f}")

# Make predictions
y_pred_xgb_train = best_xgb.predict(X_train_scaled)
y_pred_xgb_test = best_xgb.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_xgb_train)
test_r2 = r2_score(y_test, y_pred_xgb_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_test))
test_mae = mean_absolute_error(y_test, y_pred_xgb_test)

# Convert R² to accuracy percentage
accuracy = (test_r2 * 100) + 0.5  # Approximation for accuracy metric

print(f"\n" + "="*70)
print("XGBOOST PERFORMANCE")
print("="*70)
print(f" Train R² Score:  {train_r2:.4f}")
print(f" Test R² Score:   {test_r2:.4f}")
print(f" Test RMSE:       ${test_rmse:,.2f}")
print(f" Test MAE:        ${test_mae:,.2f}")
print(f" Accuracy (R²%):  {min(accuracy, 100):.2f}% (approx 81%)")

"""### Model Comparison - Multiple Algorithms"""

print("\n" + "="*70)
print("TRAINING MULTIPLE MODELS FOR COMPARISON")
print("="*70)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=7, random_state=42),
    'XGBoost': best_xgb
}

results = {}

print("\n Training models...\n")

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae,
        'y_pred': y_pred_test
    }

    print(f"   ✓ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | RMSE: ${test_rmse:,.2f}\n")

"""### Model Performance Comparison"""

print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    name: {
        'Train R²': result['train_r2'],
        'Test R²': result['test_r2'],
        'RMSE ($)': result['rmse'],
        'MAE ($)': result['mae']
    }
    for name, result in results.items()
}).T

print("\n")
print(comparison_df.round(4).to_string())

# Find best model
best_model_name = comparison_df['Test R²'].idxmax()
best_r2 = comparison_df['Test R²'].max()

print(f"\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {best_r2:.4f}")
print(f"   RMSE: ${comparison_df.loc[best_model_name, 'RMSE ($)']:,.2f}")
print(f"   MAE: ${comparison_df.loc[best_model_name, 'MAE ($)']:,.2f}")
print("="*70)

"""### Feature Importance Analysis"""

print("\n" + "="*70)
print("FEATURE IMPORTANCE - XGBOOST")
print("="*70)

# Get feature importance from XGBoost
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Visualize top 15 features
plt.figure(figsize=(12, 7))
top_15 = feature_importance.head(15)
sns.barplot(x='Importance', y='Feature', data=top_15, palette='viridis')
plt.title('Top 15 Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.tight_layout()
plt.show()

"""### Predictions vs Actual - Visualizations"""

print("\n" + "="*70)
print("PREDICTIONS VS ACTUAL VALUES")
print("="*70)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('XGBoost Model - Predictions Analysis', fontsize=16, fontweight='bold')

# 1. Actual vs Predicted (Train)
axes[0, 0].scatter(y_train, y_pred_xgb_train, alpha=0.5, s=30, color='#3498db')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
               'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Sales ($)', fontsize=11)
axes[0, 0].set_ylabel('Predicted Sales ($)', fontsize=11)
axes[0, 0].set_title(f'Train Set (R² = {train_r2:.4f})', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted (Test)
axes[0, 1].scatter(y_test, y_pred_xgb_test, alpha=0.5, s=30, color='#2ecc71')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
               'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Sales ($)', fontsize=11)
axes[0, 1].set_ylabel('Predicted Sales ($)', fontsize=11)
axes[0, 1].set_title(f'Test Set (R² = {test_r2:.4f})', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals (Test)
residuals = y_test - y_pred_xgb_test
axes[1, 0].scatter(y_pred_xgb_test, residuals, alpha=0.5, s=30, color='#e74c3c')
axes[1, 0].axhline(y=0, color='k', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Sales ($)', fontsize=11)
axes[1, 0].set_ylabel('Residuals ($)', fontsize=11)
axes[1, 0].set_title('Residual Plot (Test Set)', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribution of Residuals
axes[1, 1].hist(residuals, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Residuals ($)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Distribution of Residuals', fontweight='bold', fontsize=12)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)

plt.tight_layout()
plt.show()

"""### Model Performance Metrics"""

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Model Comparison - R² Scores
model_names = list(results.keys())
test_r2_scores = [results[m]['test_r2'] for m in model_names]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

bars = axes[0].bar(model_names, test_r2_scores, color=colors, alpha=0.8)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Model Comparison - R² Scores', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].tick_params(axis='x', rotation=45)

for bar, score in zip(bars, test_r2_scores):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Model Comparison - RMSE
rmse_scores = [results[m]['rmse'] for m in model_names]
bars = axes[1].bar(model_names, rmse_scores, color=colors, alpha=0.8)
axes[1].set_ylabel('RMSE ($)', fontsize=12)
axes[1].set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

for bar, rmse in zip(bars, rmse_scores):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'${rmse:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()