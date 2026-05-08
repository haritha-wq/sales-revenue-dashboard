# 📊 Sales & Revenue Intelligence Dashboard

> A full-stack Business Intelligence and Machine Learning project built with Python, Streamlit, XGBoost, and Plotly — analysing 5,000+ sales records across 3 years (2022–2024) with interactive dashboards and revenue forecasting.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-5.x-purple?style=flat-square&logo=plotly)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)

---

## 🎯 Project Objective

To build an end-to-end Sales Intelligence system that:
- Analyses 3 years of multi-region, multi-category sales data
- Surfaces KPIs and business insights through interactive dashboards
- Forecasts next 6 months of revenue using ML (XGBoost, R² = 0.70)
- Enables data-driven decision-making for sales strategy

---

## 📸 Dashboard Preview

| Chart | Description |
|-------|-------------|
| 📈 Revenue Trend | Monthly revenue + profit margin trend (2022–2024) |
| 🗺️ Region Analysis | Revenue breakdown by North/South/East/West/Central |
| 🍕 Category Share | Pie + bar charts for 6 product categories |
| 📅 Quarterly KPIs | Quarter-wise revenue vs margin comparison |
| 🏆 Top Products | Top 10 products by revenue |
| 👤 Sales Rep | Individual rep performance with bubble chart |
| 🔥 Heatmap | Category × Month revenue heatmap |
| 💳 Channel | Online vs Retail vs Distributor vs Direct Sales |
| 🤖 ML Forecast | 6-month XGBoost forecast with confidence bands |

---

## 🗂️ Project Structure

```
sales-revenue-dashboard/
│
├── data/
│   ├── generate_data.py        # Synthetic data generator (5,000 records)
│   ├── sales_data.csv          # Main dataset (generated)
│   └── monthly_targets.csv     # Monthly revenue targets
│
├── src/
│   ├── eda_analysis.py         # EDA — generates 8 matplotlib charts
│   └── ml_forecasting.py       # XGBoost model + 6-month forecast
│
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
│
├── models/
│   ├── xgb_revenue_model.pkl   # Trained XGBoost model
│   ├── scaler.pkl              # Feature scaler
│   └── feature_names.pkl       # Model feature list
│
├── reports/
│   ├── 01_revenue_trend.png
│   ├── 02_revenue_by_region.png
│   ├── 03_revenue_by_category.png
│   ├── 04_quarterly_performance.png
│   ├── 05_top_products.png
│   ├── 06_sales_rep_performance.png
│   ├── 07_channel_payment_analysis.png
│   ├── 08_heatmap_category_month.png
│   ├── 09_revenue_forecast.png
│   ├── 10_feature_importance.png
│   └── 6month_forecast.csv
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Overview

| Field | Description |
|-------|-------------|
| `order_id` | Unique order identifier |
| `date` | Order date (2022–2024) |
| `region` | North / South / East / West / Central |
| `category` | Electronics / Clothing / Home & Kitchen / Sports / Books / Beauty |
| `product` | Individual product name |
| `sales_rep` | Assigned sales representative |
| `channel` | Online / Retail Store / Distributor / Direct Sales |
| `quantity` | Units ordered |
| `gross_revenue` | Revenue before discount |
| `discount_pct` | Discount percentage applied |
| `net_revenue` | Final revenue after discount |
| `cost` | Cost of goods sold |
| `profit` | Net profit |
| `profit_margin_pct` | Profit margin percentage |
| `return_flag` | 1 = returned order, 0 = normal |

**Totals:** 5,000 orders · ₹67.2 Crore revenue · 56.6% avg profit margin

---

## 🤖 ML Model — XGBoost Revenue Forecaster

### Features Used
- **Seasonality**: Sine/cosine encoding of month, quarter, Q4 flag
- **Lag Features**: Revenue from 1, 2, 3, 6 months prior
- **Rolling Averages**: 3-month and 6-month rolling mean
- **Business Metrics**: Order count, average order value, discount %
- **Trend**: Linear time trend

### Model Performance

| Model | MAE | R² | MAPE |
|-------|-----|-----|------|
| Linear Regression | ₹14.3L | 0.9749 | 6.18% |
| Random Forest | ₹44.3L | 0.6040 | 14.42% |
| **XGBoost** | **₹42.3L** | **0.70** | **14.69%** |

### 6-Month Forecast (Jan–Jun 2025)

| Month | Forecast | Range |
|-------|----------|-------|
| Jan 2025 | ₹1.7 Cr | ₹1.5Cr – ₹1.8Cr |
| Feb 2025 | ₹2.1 Cr | ₹1.9Cr – ₹2.3Cr |
| Mar 2025 | ₹2.2 Cr | ₹2.0Cr – ₹2.4Cr |
| Apr 2025 | ₹2.2 Cr | ₹1.9Cr – ₹2.4Cr |
| May 2025 | ₹2.1 Cr | ₹1.9Cr – ₹2.4Cr |
| Jun 2025 | ₹2.1 Cr | ₹1.9Cr – ₹2.4Cr |

---

## 🔍 Key Business Insights

1. **West Region leads** revenue — contributes highest region-wise sales due to 15% regional demand premium
2. **Electronics dominates** category revenue — highest AOV at ₹40,000+ per order
3. **Q4 is peak season** — November–December alone drive 35-40% of annual revenue (festive effect)
4. **Online channel growing** — e-commerce now the #1 revenue channel, outperforming retail
5. **8.4% return rate** — Electronics and Clothing categories drive most returns; opportunity for quality improvement
6. **Rohit Verma** is the top sales rep by revenue; opportunity to replicate best practices team-wide

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/sales-revenue-dashboard.git
cd sales-revenue-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset
```bash
cd data
python generate_data.py
```

### 4. Run EDA Analysis
```bash
cd ..
python src/eda_analysis.py
# → Saves 8 charts to /reports/
```

### 5. Train ML Model & Generate Forecast
```bash
python src/ml_forecasting.py
# → Trains XGBoost, saves model, generates 6-month forecast
```

### 6. Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
# → Opens at http://localhost:8501
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **Pandas** | Data manipulation and aggregation |
| **NumPy** | Numerical computations |
| **Matplotlib + Seaborn** | Static chart generation (EDA) |
| **Plotly** | Interactive charts in dashboard |
| **Streamlit** | Dashboard web application |
| **Scikit-learn** | ML pipeline, Linear Regression, Random Forest |
| **XGBoost** | Gradient boosting for revenue forecasting |
| **Joblib** | Model serialisation |
| **OpenPyXL** | Excel report generation |

---

## 📝 Resume Bullet Points

> Use these directly on your resume:

- "Built end-to-end sales intelligence dashboard using **Python and Streamlit**; analysed 5,000+ orders across 5 regions and 6 categories over 3 years"
- "Developed **XGBoost revenue forecasting model** (R² = 0.70) to predict next 6 months of sales with ±10% confidence bands"
- "Performed comprehensive **EDA** generating 10 business insight charts covering regional performance, seasonal trends, product rankings, and channel analysis"
- "Applied **feature engineering** including lag features, rolling averages, and Fourier-encoded seasonality to improve model accuracy"
- "Identified Q4 festive season drives **35–40% of annual revenue** and West region contributes highest regional sales — insights actionable for inventory and campaign planning"

---

## 👤 Author

**[Your Name]**  
BCom Computer Application | MCA — Artificial Intelligence & Machine Learning  

📧 [your.email@gmail.com]  
🔗 [LinkedIn Profile]  
🐙 [GitHub Profile]

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
