---

## Global Sales Prediction and Analysis using XGBoost & Power BI

### 📌 Project Overview

This project builds a **data-driven sales prediction system** using **XGBoost** and other regression algorithms.
It performs **data cleaning, feature engineering, EDA, model training, and evaluation**, followed by **business intelligence visualization in Power BI**.

The goal is to **predict future sales** and uncover **key drivers influencing sales performance** — helping businesses optimize pricing, forecast demand, and improve decision-making.

---

## ⚙️ Features

✅ Automated data cleaning & preprocessing
✅ Exploratory Data Analysis (EDA) with rich visualizations
✅ Advanced feature engineering (price, customer, product, and temporal features)
✅ Model training & comparison (Linear, Ridge, RF, GB, XGBoost)
✅ Hyperparameter tuning with GridSearchCV
✅ Performance evaluation (R², RMSE, MAE)
✅ Feature importance analysis (Top 15 predictive factors)
✅ Integration with **Power BI dashboards** for interactive reporting

---

## 📂 Project Structure

```
📁 Sales_Prediction_Project/
│
├── sales_data.csv                  # Raw dataset
├── sales_prediction_model.py       # Main Python script
├── Telecom_ChurnCleaned.csv        # (Optional) preprocessed dataset
├── README.md                       # Project documentation
├── powerbi_dashboard.pbix          # Power BI visualization dashboard
```

---

## 🧰 Requirements

| Library                 | Description                              |
| ----------------------- | ---------------------------------------- |
| `pandas`, `numpy`       | Data manipulation & numerical operations |
| `matplotlib`, `seaborn` | Visualization                            |
| `scikit-learn`          | Preprocessing, training & metrics        |
| `xgboost`               | Gradient boosting regressor              |
| `warnings`              | Handling warnings gracefully             |

---

## 🖥️ Installation

```bash
pip install xgboost pandas scikit-learn matplotlib seaborn numpy
```

---

## 🧮 Workflow Summary

### 1️⃣ Data Loading & Cleaning

* Import `sales_data.csv`
* Drop irrelevant columns (`ADDRESSLINE2`, `ADDRESSLINE3`, etc.)
* Handle missing values via median imputation
* Remove null postal codes
* Rename columns for clarity (e.g., `ADDRESSLINE1 → ADDRESS`)

---

### 2️⃣ Exploratory Data Analysis (EDA)

Comprehensive visual analysis using `matplotlib` and `seaborn`:

* Sales distribution histogram
* Total sales by product line
* Average sales by year and quarter
* Quantity vs. Sales correlation
* Total sales by order status

🖼️ Example visuals:

* `Sales by Product Line`
* `Sales by Year`
* `Quantity Ordered vs Sales`

---

### 3️⃣ Feature Engineering

Creation of **derived & aggregated features** to boost predictive power:

* **Price-based:** `PROFIT_MARGIN`, `PRICE_DISCOUNT`, `REVENUE_PER_UNIT`
* **Temporal:** `DAY_OF_WEEK`, `DAY_OF_MONTH`, `WEEK_OF_YEAR`
* **Customer-level:** total and average sales per customer
* **Product-level:** average and total sales per product line
* **Interaction:** `QTY_PRICE_INTERACTION`, `STATUS_MSRP_RATIO`

---

### 4️⃣ Model Preparation

* Encode categorical variables using `LabelEncoder`
* Split dataset: 80% train / 20% test
* Apply `StandardScaler` for numerical normalization

---

### 5️⃣ Model Training & Tuning

#### 🧠 Algorithms:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* **XGBoost Regressor (Tuned)**

#### 🎯 Hyperparameter Optimization:

Grid search with:

```python
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.1]
}
```

Best model achieved:

* **R² (Test): ≈ 0.81**
* **RMSE: Low error variance**
* **MAE: Stable across predictions**

---

### 6️⃣ Model Evaluation

Metrics:

* **R² Score**
* **RMSE (Root Mean Square Error)**
* **MAE (Mean Absolute Error)**

Comparison table printed as:

```
| Model              | Train R² | Test R² | RMSE ($) | MAE ($) |
|--------------------|----------|---------|-----------|---------|
| Linear Regression  | 0.73     | 0.70    | 1750.32   | 1210.21 |
| XGBoost (Tuned)    | 0.86     | 0.81    | 1250.10   | 920.55  |
```

---

### 7️⃣ Feature Importance

Top predictive factors identified by XGBoost:

1. `PRICEEACH`
2. `QUANTITYORDERED`
3. `CUSTOMER_TOTAL_SALES`
4. `REVENUE_PER_UNIT`
5. `PRODUCTLINE_TOTAL_SALES`

Bar chart generated for top 15 features.

---

### 8️⃣ Predictions Visualization

Includes:

* Actual vs. Predicted (Train/Test)
* Residual distribution
* Model comparison (R² & RMSE)

---

## 📊 Power BI Integration

After model training, the processed dataset and predictions are exported for visualization.

### Steps:

1. Export model predictions:

   ```python
   predictions_df = pd.DataFrame({
       'Actual_Sales': y_test,
       'Predicted_Sales': y_pred_xgb_test
   })
   predictions_df.to_csv('sales_predictions_output.csv', index=False)
   ```

2. Import `sales_predictions_output.csv` into Power BI.

3. Create the following visuals:

   * 📈 **Actual vs Predicted Sales (Scatter)**
   * 📊 **Top 10 Customers by Sales**
   * 💡 **Feature Importance Breakdown**
   * 🌍 **Sales by Country & Product Line (Map)**
   * ⏳ **Sales Trend by Year & Quarter**

4. Publish the Power BI dashboard (`powerbi_dashboard.pbix`) for stakeholders.

---

## 📈 Results Summary

* Best Model: **XGBoost**
* Test R² ≈ **0.81**
* RMSE: **Low deviation from actuals**
* Feature insights aligned with sales patterns (pricing, quantity, customer loyalty)
* Power BI visualizations deliver business interpretability

---

## 🚀 Future Enhancements

* Integrate with real-time sales API (e.g., Salesforce)
* Deploy model via Flask or FastAPI
* Automate data refresh and Power BI update
* Add time-series forecasting (Prophet or LSTM)

---