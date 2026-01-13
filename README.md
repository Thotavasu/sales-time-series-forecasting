# 📈 Monthly Sales Forecasting – Retail Business Case

## Overview
This project demonstrates an **end-to-end time series forecasting pipeline** to predict **monthly retail sales** using historical transaction data.

The objective is to help a retail business make **data-driven decisions** for:
- Inventory planning
- Staffing allocation
- Financial forecasting
- Supply chain operations

The project is designed to reflect **real-world data science workflows** used in industry.

---

## Business Problem
Retail companies need accurate sales forecasts to avoid:
- Overstocking (high holding costs)
- Understocking (lost revenue)
- Poor staffing decisions during peak seasons

Using historical sales data, we forecast **future monthly sales** to support better operational planning.

---

## Dataset
- Source: Kaggle – Superstore Sales Dataset
- Type: Transaction-level retail data
- Key fields used:
  - Order Date
  - Sales

The data is aggregated into **monthly sales totals**, which is the most common granularity for business planning.

---

## Methodology
1. **Data Preparation**
   - Converted order dates to datetime
   - Aggregated transaction data into monthly sales

2. **Exploratory Data Analysis**
   - Identified trend and seasonality
   - Observed strong sales peaks in Q4 (Nov–Dec)

3. **Modeling**
   - SARIMA (statistical baseline)
   - Prophet (trend + seasonality focused)

4. **Evaluation**
   - Time-based 80/20 train-test split
   - Metrics used:
     - MAE
     - RMSE
     - MAPE

5. **Forecasting**
   - Generated 6-month and 12-month future sales forecasts
   - Visualized confidence intervals for risk-aware planning

---

## Key Insights
- Sales show **clear yearly seasonality**
- Strong and consistent demand peaks during **November–December**
- Prophet performed better due to explicit handling of trend and seasonality

---

## Business Recommendations
- Increase inventory levels ahead of Q4
- Plan additional staffing during peak months
- Use forecast confidence intervals to prepare best- and worst-case scenarios

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib
- Statsmodels (SARIMA)
- Prophet
- Scikit-learn
- Git & GitHub
- Jupyter Notebook

---

## How to Run the Project
```bash
pip install -r requirements.txt
python -m src.prepare_data
python -m src.evaluate
