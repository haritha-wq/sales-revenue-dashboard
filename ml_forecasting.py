"""
Sales & Revenue Intelligence Dashboard
ML Revenue Forecasting Model — XGBoost + Linear Regression

Predicts next 6 months of revenue.
Saves model artifacts to /models/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../models', exist_ok=True)
os.makedirs('../reports', exist_ok=True)

BG_COLOR   = '#0F1117'
TEXT_COLOR  = '#E8E8E8'
plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor':   '#1A1D27',
    'axes.edgecolor':   '#2E3140',
    'axes.labelcolor':  TEXT_COLOR,
    'xtick.color':      TEXT_COLOR,
    'ytick.color':      TEXT_COLOR,
    'text.color':       TEXT_COLOR,
    'grid.color':       '#2E3140',
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

def fmt_inr(x, pos=None):
    if x >= 1e7: return f'₹{x/1e7:.1f}Cr'
    if x >= 1e5: return f'₹{x/1e5:.1f}L'
    return f'₹{x:,.0f}'

# ─── Load & Aggregate ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('../data/sales_data.csv', parse_dates=['date'])

monthly = df.groupby(['year', 'month_num']).agg(
    revenue=('net_revenue', 'sum'),
    orders=('order_id', 'count'),
    avg_order=('net_revenue', 'mean'),
    profit=('profit', 'sum'),
    discount_avg=('discount_pct', 'mean'),
    quantity=('quantity', 'sum'),
).reset_index()

monthly['period'] = pd.to_datetime(monthly[['year', 'month_num']].rename(
    columns={'year': 'year', 'month_num': 'month'}).assign(day=1))
monthly = monthly.sort_values('period').reset_index(drop=True)

# ─── Feature Engineering ──────────────────────────────────────────────────────
print("Engineering features...")
monthly['month_sin'] = np.sin(2 * np.pi * monthly['month_num'] / 12)
monthly['month_cos'] = np.cos(2 * np.pi * monthly['month_num'] / 12)
monthly['quarter']   = ((monthly['month_num'] - 1) // 3) + 1
monthly['is_q4']     = (monthly['quarter'] == 4).astype(int)
monthly['trend']     = np.arange(len(monthly))

# Lag features
for lag in [1, 2, 3, 6]:
    monthly[f'lag_{lag}'] = monthly['revenue'].shift(lag)

# Rolling stats
monthly['rolling_3m_avg'] = monthly['revenue'].shift(1).rolling(3).mean()
monthly['rolling_6m_avg'] = monthly['revenue'].shift(1).rolling(6).mean()
monthly['mom_growth']     = monthly['revenue'].pct_change(1)
monthly['yoy_growth']     = monthly['revenue'].pct_change(12)

monthly = monthly.dropna().reset_index(drop=True)
print(f"  Feature matrix: {monthly.shape}")

# ─── Model Training ───────────────────────────────────────────────────────────
FEATURES = ['month_sin', 'month_cos', 'quarter', 'is_q4', 'trend',
            'lag_1', 'lag_2', 'lag_3', 'lag_6',
            'rolling_3m_avg', 'rolling_6m_avg',
            'orders', 'avg_order', 'discount_avg']

X = monthly[FEATURES]
y = monthly['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"\nTrain: {len(X_train)} months | Test: {len(X_test)} months")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
    'XGBoost':           xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                           max_depth=5, subsample=0.8,
                                           colsample_bytree=0.8, random_state=42,
                                           verbosity=0),
}

results = {}
print("\nModel Evaluation:")
print("-" * 55)
for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
                     'y_pred': y_pred, 'model': model}
    print(f"  {name:<22} | MAE: {fmt_inr(mae):<12} | R²: {r2:.4f} | MAPE: {mape:.2f}%")

# ─── Best Model: XGBoost ──────────────────────────────────────────────────────
best_name = 'XGBoost'
best_model  = results[best_name]['model']
best_scaler = scaler

joblib.dump(best_model, '../models/xgb_revenue_model.pkl')
joblib.dump(scaler,     '../models/scaler.pkl')
joblib.dump(FEATURES,   '../models/feature_names.pkl')
print(f"\n  ✓ Best model ({best_name}) saved to /models/")

# ─── 6-Month Forecast ─────────────────────────────────────────────────────────
print("\nGenerating 6-month forecast...")
last_row  = monthly.iloc[-1]
last_date = last_row['period']

forecast_rows = []
current_monthly = monthly.copy()

for i in range(1, 7):
    next_date   = last_date + pd.DateOffset(months=i)
    next_month  = next_date.month
    next_year   = next_date.year
    next_trend  = last_row['trend'] + i

    row = {
        'month_sin':      np.sin(2 * np.pi * next_month / 12),
        'month_cos':      np.cos(2 * np.pi * next_month / 12),
        'quarter':        ((next_month - 1) // 3) + 1,
        'is_q4':          int(((next_month - 1) // 3) + 1 == 4),
        'trend':          next_trend,
        'lag_1':          current_monthly['revenue'].iloc[-1],
        'lag_2':          current_monthly['revenue'].iloc[-2],
        'lag_3':          current_monthly['revenue'].iloc[-3],
        'lag_6':          current_monthly['revenue'].iloc[-6],
        'rolling_3m_avg': current_monthly['revenue'].iloc[-3:].mean(),
        'rolling_6m_avg': current_monthly['revenue'].iloc[-6:].mean(),
        'orders':         current_monthly['orders'].mean(),
        'avg_order':      current_monthly['avg_order'].mean(),
        'discount_avg':   current_monthly['discount_avg'].mean(),
    }

    X_fore = pd.DataFrame([row])[FEATURES]
    pred   = best_model.predict(X_fore)[0]

    # Confidence interval ±10%
    lower = pred * 0.90
    upper = pred * 1.10

    forecast_rows.append({
        'period': next_date,
        'forecast_revenue': round(pred, 2),
        'lower_bound':      round(lower, 2),
        'upper_bound':      round(upper, 2),
    })

    # Add prediction back for next iteration
    new_row = current_monthly.iloc[-1:].copy()
    new_row['period']    = next_date
    new_row['revenue']   = pred
    new_row['month_num'] = next_month
    new_row['trend']     = next_trend
    current_monthly = pd.concat([current_monthly, new_row], ignore_index=True)

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv('../reports/6month_forecast.csv', index=False)

print("\n  6-Month Revenue Forecast:")
print("  " + "-"*48)
for _, r in forecast_df.iterrows():
    month_label = r['period'].strftime('%b %Y')
    print(f"  {month_label:<10} | {fmt_inr(r['forecast_revenue']):<12} | [{fmt_inr(r['lower_bound'])} – {fmt_inr(r['upper_bound'])}]")

# ─── Forecast Chart ───────────────────────────────────────────────────────────
print("\nSaving forecast chart...")
fig, ax = plt.subplots(figsize=(14, 6))

# Actual data (last 18 months)
hist = monthly.tail(18)
ax.plot(hist['period'], hist['revenue'], color='#2D4CFF', linewidth=2.5, label='Actual Revenue')
ax.fill_between(hist['period'], hist['revenue'], alpha=0.1, color='#2D4CFF')

# Test predictions
test_periods = monthly.iloc[len(X_train):]['period']
ax.plot(test_periods, results[best_name]['y_pred'], '--', color='#FFD60A',
        linewidth=1.8, label='Model Fit (Test)', alpha=0.8)

# Forecast
ax.plot(forecast_df['period'], forecast_df['forecast_revenue'],
        color='#1DB954', linewidth=2.5, marker='o', markersize=7, label='6M Forecast')
ax.fill_between(forecast_df['period'],
                forecast_df['lower_bound'], forecast_df['upper_bound'],
                alpha=0.15, color='#1DB954', label='Confidence Band (±10%)')

# Annotate forecast values
for _, row in forecast_df.iterrows():
    ax.annotate(fmt_inr(row['forecast_revenue']),
                xy=(row['period'], row['forecast_revenue']),
                xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=8, color='#1DB954')

ax.axvline(x=monthly['period'].iloc[-1], color='#FF6B35', linestyle=':', linewidth=1.5, label='Forecast Start')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
ax.set_title(f'Revenue Forecast — Next 6 Months  |  {best_name} (R² = {results[best_name]["r2"]:.3f})',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month')
ax.set_ylabel('Revenue')
ax.legend(loc='upper left', framealpha=0.2, fontsize=9)
ax.grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.savefig('../reports/09_revenue_forecast.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Feature Importance ───────────────────────────────────────────────────────
fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(10)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(fi.index, fi.values, color='#2D4CFF', height=0.6)
ax.set_title('XGBoost Feature Importance (Top 10)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Importance Score')
for bar, val in zip(bars, fi.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../reports/10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n  ✓ Forecast chart saved to /reports/09_revenue_forecast.png")
print("  ✓ Feature importance saved to /reports/10_feature_importance.png")
print("\n  Model training complete!")
