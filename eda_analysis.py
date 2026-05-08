"""
Sales & Revenue Intelligence Dashboard
Exploratory Data Analysis (EDA) Script

Run this script to generate all analysis charts and insights.
Output: /reports/ folder with PNG charts + summary Excel report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Setup ────────────────────────────────────────────────────────────────────
os.makedirs('../reports', exist_ok=True)

PALETTE   = ['#2D4CFF', '#FF6B35', '#1DB954', '#FFD60A', '#E040FB', '#00BCD4']
BG_COLOR  = '#0F1117'
TEXT_COLOR = '#E8E8E8'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor':   '#1A1D27',
    'axes.edgecolor':   '#2E3140',
    'axes.labelcolor':  TEXT_COLOR,
    'xtick.color':      TEXT_COLOR,
    'ytick.color':      TEXT_COLOR,
    'text.color':       TEXT_COLOR,
    'grid.color':       '#2E3140',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

def fmt_inr(x, pos=None):
    if x >= 1e7:
        return f'₹{x/1e7:.1f}Cr'
    elif x >= 1e5:
        return f'₹{x/1e5:.1f}L'
    return f'₹{x:,.0f}'

# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('sales_data.csv', parse_dates=['date'])
targets = pd.read_csv('monthly_targets.csv')
print(f"  Loaded {len(df):,} records | {df['date'].min().date()} → {df['date'].max().date()}")

# ─── 1. Revenue Trend Over Time ───────────────────────────────────────────────
print("\n[1/8] Revenue Trend...")
monthly = df.groupby(['year', 'month_num'])['net_revenue'].sum().reset_index()
monthly['period'] = pd.to_datetime(monthly[['year','month_num']].rename(columns={'year':'year','month_num':'month'}).assign(day=1))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly['period'], monthly['net_revenue'], color='#2D4CFF', linewidth=2.5, zorder=3)
ax.fill_between(monthly['period'], monthly['net_revenue'], alpha=0.15, color='#2D4CFF')
ax.scatter(monthly['period'], monthly['net_revenue'], color='#2D4CFF', s=50, zorder=4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
ax.set_title('Monthly Revenue Trend (2022–2024)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month')
ax.set_ylabel('Net Revenue')
ax.grid(axis='y')
# Annotate peak
peak_idx = monthly['net_revenue'].idxmax()
ax.annotate(f"Peak: {fmt_inr(monthly.loc[peak_idx,'net_revenue'])}",
            xy=(monthly.loc[peak_idx,'period'], monthly.loc[peak_idx,'net_revenue']),
            xytext=(0, 15), textcoords='offset points',
            ha='center', fontsize=9, color='#FFD60A',
            arrowprops=dict(arrowstyle='->', color='#FFD60A', lw=1.5))
plt.tight_layout()
plt.savefig('../reports/01_revenue_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 2. Revenue by Region ─────────────────────────────────────────────────────
print("[2/8] Revenue by Region...")
region_rev = df.groupby('region')['net_revenue'].sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(region_rev.index, region_rev.values, color=PALETTE[:len(region_rev)], height=0.55)
for bar, val in zip(bars, region_rev.values):
    ax.text(val + region_rev.max()*0.01, bar.get_y() + bar.get_height()/2,
            fmt_inr(val), va='center', fontsize=10, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
ax.set_title('Net Revenue by Region', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Net Revenue')
plt.tight_layout()
plt.savefig('../reports/02_revenue_by_region.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 3. Revenue by Category ───────────────────────────────────────────────────
print("[3/8] Revenue by Category...")
cat_rev = df.groupby('category').agg(
    revenue=('net_revenue', 'sum'),
    profit=('profit', 'sum'),
    orders=('order_id', 'count')
).reset_index().sort_values('revenue', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1, ax2 = axes

bars = ax1.bar(cat_rev['category'], cat_rev['revenue'], color=PALETTE, width=0.6)
ax1.set_title('Revenue by Category', fontsize=13, fontweight='bold')
ax1.set_ylabel('Net Revenue')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
ax1.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, cat_rev['revenue']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cat_rev['revenue'].max()*0.01,
             fmt_inr(val), ha='center', fontsize=8)

wedges, texts, autotexts = ax2.pie(cat_rev['revenue'], labels=cat_rev['category'],
                                    colors=PALETTE, autopct='%1.1f%%',
                                    startangle=90, pctdistance=0.75,
                                    wedgeprops={'edgecolor': BG_COLOR, 'linewidth': 2})
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')
ax2.set_title('Revenue Share by Category', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/03_revenue_by_category.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 4. Quarterly Performance ─────────────────────────────────────────────────
print("[4/8] Quarterly Performance...")
q_perf = df.groupby(['year','quarter']).agg(
    revenue=('net_revenue','sum'),
    profit=('profit','sum'),
    orders=('order_id','count')
).reset_index()
q_perf['label'] = q_perf['year'].astype(str) + '-' + q_perf['quarter']
q_perf['profit_margin'] = (q_perf['profit'] / q_perf['revenue'] * 100).round(1)

fig, ax1 = plt.subplots(figsize=(14, 5))
x = range(len(q_perf))
bars = ax1.bar(x, q_perf['revenue'], color='#2D4CFF', alpha=0.8, width=0.6, label='Revenue')
ax1.set_xticks(x)
ax1.set_xticklabels(q_perf['label'], rotation=30)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
ax1.set_ylabel('Net Revenue')
ax1.set_title('Quarterly Revenue & Profit Margin (2022–2024)', fontsize=14, fontweight='bold', pad=15)

ax2 = ax1.twinx()
ax2.plot(x, q_perf['profit_margin'], color='#FFD60A', linewidth=2.5, marker='D',
         markersize=7, label='Profit Margin %', zorder=5)
ax2.set_ylabel('Profit Margin (%)', color='#FFD60A')
ax2.tick_params(axis='y', labelcolor='#FFD60A')
ax2.set_ylim(0, 100)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.2)
plt.tight_layout()
plt.savefig('../reports/04_quarterly_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 5. Top 10 Products ───────────────────────────────────────────────────────
print("[5/8] Top Products...")
top_products = df.groupby('product')['net_revenue'].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
colors = [PALETTE[0]] * 3 + [PALETTE[1]] * 3 + [PALETTE[2]] * 4
bars = ax.barh(top_products.index[::-1], top_products.values[::-1], color=colors[::-1], height=0.55)
for bar, val in zip(bars, top_products.values[::-1]):
    ax.text(val + top_products.max()*0.005, bar.get_y() + bar.get_height()/2,
            fmt_inr(val), va='center', fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
ax.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Net Revenue')
plt.tight_layout()
plt.savefig('../reports/05_top_products.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 6. Sales Rep Performance ─────────────────────────────────────────────────
print("[6/8] Sales Rep Performance...")
rep_perf = df.groupby('sales_rep').agg(
    revenue=('net_revenue','sum'),
    orders=('order_id','count'),
    profit=('profit','sum')
).reset_index().sort_values('revenue', ascending=False)
rep_perf['avg_order'] = (rep_perf['revenue'] / rep_perf['orders']).round(0)
rep_perf['profit_margin'] = (rep_perf['profit'] / rep_perf['revenue'] * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
c = [PALETTE[0] if i < 3 else '#3A3F55' for i in range(len(rep_perf))]
bars = axes[0].barh(rep_perf['sales_rep'][::-1], rep_perf['revenue'][::-1], color=c[::-1], height=0.55)
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
axes[0].set_title('Revenue by Sales Rep', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Net Revenue')
for bar, val in zip(bars, rep_perf['revenue'][::-1]):
    axes[0].text(val + rep_perf['revenue'].max()*0.005, bar.get_y() + bar.get_height()/2,
                 fmt_inr(val), va='center', fontsize=8)

rep_colors = (PALETTE * ((len(rep_perf) // len(PALETTE)) + 1))[:len(rep_perf)]
axes[1].scatter(rep_perf['orders'], rep_perf['profit_margin'],
                s=rep_perf['revenue']/1e5, c=rep_colors, alpha=0.8, edgecolors='white', linewidth=0.5)
for _, row in rep_perf.iterrows():
    axes[1].annotate(row['sales_rep'].split()[0],
                     (row['orders'], row['profit_margin']),
                     textcoords='offset points', xytext=(5, 5), fontsize=8)
axes[1].set_xlabel('Number of Orders')
axes[1].set_ylabel('Profit Margin (%)')
axes[1].set_title('Orders vs Profit Margin (bubble = revenue)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/06_sales_rep_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 7. Channel & Payment Analysis ───────────────────────────────────────────
print("[7/8] Channel Analysis...")
ch_rev = df.groupby('channel')['net_revenue'].sum().sort_values(ascending=False)
pay_rev = df.groupby('payment_method')['net_revenue'].sum().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(ch_rev.index, ch_rev.values, color=PALETTE[:len(ch_rev)], width=0.5)
axes[0].set_title('Revenue by Sales Channel', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Net Revenue')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
for i, (idx, val) in enumerate(ch_rev.items()):
    axes[0].text(i, val + ch_rev.max()*0.01, fmt_inr(val), ha='center', fontsize=9)

wedges, _, autotexts = axes[1].pie(pay_rev.values, labels=pay_rev.index,
                                    colors=PALETTE, autopct='%1.1f%%',
                                    startangle=90, pctdistance=0.78,
                                    wedgeprops={'edgecolor': BG_COLOR, 'linewidth': 2})
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')
axes[1].set_title('Payment Method Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/07_channel_payment_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 8. Heatmap: Month × Category Revenue ────────────────────────────────────
print("[8/8] Heatmap...")
heat_data = df.pivot_table(values='net_revenue', index='category', columns='month_num', aggfunc='sum')
heat_data.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heat_data/1e5, annot=True, fmt='.0f', cmap='Blues',
            ax=ax, linewidths=0.5, linecolor='#0F1117',
            cbar_kws={'label': 'Revenue (₹ Lakhs)'})
ax.set_title('Category × Month Revenue Heatmap (₹ Lakhs)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month')
ax.set_ylabel('Category')
plt.tight_layout()
plt.savefig('../reports/08_heatmap_category_month.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Summary KPIs ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  BUSINESS INTELLIGENCE SUMMARY")
print("="*50)
print(f"  Total Revenue   : {fmt_inr(df['net_revenue'].sum())}")
print(f"  Total Profit    : {fmt_inr(df['profit'].sum())}")
print(f"  Avg Profit Mgn  : {df['profit_margin_pct'].mean():.1f}%")
print(f"  Total Orders    : {len(df):,}")
print(f"  Avg Order Value : {fmt_inr(df['net_revenue'].mean())}")
print(f"  Return Rate     : {df['return_flag'].mean()*100:.1f}%")
print(f"  Top Region      : {df.groupby('region')['net_revenue'].sum().idxmax()}")
print(f"  Top Category    : {df.groupby('category')['net_revenue'].sum().idxmax()}")
print(f"  Top Product     : {df.groupby('product')['net_revenue'].sum().idxmax()}")
print(f"  Top Sales Rep   : {df.groupby('sales_rep')['net_revenue'].sum().idxmax()}")
print("="*50)
print("\n  ✓ All 8 charts saved to /reports/")
