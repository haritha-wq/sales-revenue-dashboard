"""
Sales & Revenue Intelligence Dashboard
Interactive Streamlit App

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d27 0%, #12151f 100%);
        border: 1px solid #2e3140;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.78rem; color: #8b92a5; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #e8e8e8; }
    .metric-delta { font-size: 0.8rem; margin-top: 4px; }
    .positive { color: #1db954; }
    .negative { color: #ff4444; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #e8e8e8;
        border-left: 3px solid #2D4CFF; padding-left: 12px;
        margin: 1.5rem 0 1rem;
    }
    .stSelectbox > div > div { background: #1a1d27; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, 'data', 'sales_data.csv'), parse_dates=['date'])
    targets = pd.read_csv(os.path.join(base, 'data', 'monthly_targets.csv'))
    return df, targets

@st.cache_resource
def load_model():
    try:
        base   = os.path.dirname(__file__)
        model  = joblib.load(os.path.join(base, 'models', 'xgb_revenue_model.pkl'))
        scaler = joblib.load(os.path.join(base, 'models', 'scaler.pkl'))
        feats  = joblib.load(os.path.join(base, 'models', 'feature_names.pkl'))
        return model, scaler, feats
    except:
        return None, None, None

df, targets = load_data()
model, scaler, feature_names = load_model()

def fmt_inr(val):
    if val >= 1e7:  return f"₹{val/1e7:.2f} Cr"
    if val >= 1e5:  return f"₹{val/1e5:.2f} L"
    if val >= 1000: return f"₹{val/1e3:.1f} K"
    return f"₹{val:,.0f}"

COLORS = ['#2D4CFF', '#FF6B35', '#1DB954', '#FFD60A', '#E040FB', '#00BCD4']
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a1d27',
        font=dict(color='#e8e8e8', family='Inter'),
        xaxis=dict(gridcolor='#2e3140', showgrid=True),
        yaxis=dict(gridcolor='#2e3140', showgrid=True),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#2e3140'),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔎 Filters")
    st.markdown("---")

    years = sorted(df['year'].unique())
    sel_years = st.multiselect("Year", years, default=years)

    regions = sorted(df['region'].unique())
    sel_regions = st.multiselect("Region", regions, default=regions)

    categories = sorted(df['category'].unique())
    sel_cats = st.multiselect("Category", categories, default=categories)

    channels = sorted(df['channel'].unique())
    sel_channels = st.multiselect("Sales Channel", channels, default=channels)

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.date_input("Select Range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)

    st.markdown("---")
    st.markdown("### 👤 About")
    st.markdown("""
    **Sales Revenue Intelligence Dashboard**
    
    Built with Python, Streamlit & XGBoost
    
    Data: 5,000 orders | 2022–2024
    
    *BCom CA + MCA AIML Project*
    """)

# ─── Filter Data ──────────────────────────────────────────────────────────────
mask = (
    df['year'].isin(sel_years) &
    df['region'].isin(sel_regions) &
    df['category'].isin(sel_cats) &
    df['channel'].isin(sel_channels)
)
if len(date_range) == 2:
    mask &= (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])

fdf = df[mask].copy()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 📊 Sales & Revenue Intelligence Dashboard")
st.markdown(f"Showing **{len(fdf):,}** orders · **{fdf['date'].min().strftime('%d %b %Y')}** to **{fdf['date'].max().strftime('%d %b %Y')}**")
st.markdown("---")

if fdf.empty:
    st.warning("No data for selected filters. Please adjust your selections.")
    st.stop()

# ─── KPI Metrics ──────────────────────────────────────────────────────────────
total_rev    = fdf['net_revenue'].sum()
total_profit = fdf['profit'].sum()
avg_margin   = fdf['profit_margin_pct'].mean()
total_orders = len(fdf)
avg_order    = fdf['net_revenue'].mean()
return_rate  = fdf['return_flag'].mean() * 100

# YoY comparison (if multiple years)
prev_rev = df[df['year'] == (max(sel_years)-1)]['net_revenue'].sum() if len(sel_years) > 0 else 0
yoy_growth = ((total_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)

def metric_card(col, label, value, delta=None, delta_label=""):
    delta_html = ""
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        cls  = "positive" if delta >= 0 else "negative"
        delta_html = f'<div class="metric-delta {cls}">{sign}{delta:.1f}% {delta_label}</div>'
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

metric_card(col1, "Total Revenue",    fmt_inr(total_rev),    yoy_growth, "YoY")
metric_card(col2, "Total Profit",     fmt_inr(total_profit), None)
metric_card(col3, "Avg Profit Margin", f"{avg_margin:.1f}%", None)
metric_card(col4, "Total Orders",     f"{total_orders:,}",   None)
metric_card(col5, "Avg Order Value",  fmt_inr(avg_order),    None)
metric_card(col6, "Return Rate",      f"{return_rate:.1f}%", None)

st.markdown("---")

# ─── Revenue Trend ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Revenue Trend</div>', unsafe_allow_html=True)

monthly_rev = fdf.groupby(['year','month_num']).agg(
    revenue=('net_revenue','sum'),
    orders=('order_id','count'),
    profit=('profit','sum')
).reset_index()
monthly_rev['period'] = pd.to_datetime(
    monthly_rev[['year','month_num']].rename(columns={'year':'year','month_num':'month'}).assign(day=1))
monthly_rev = monthly_rev.sort_values('period')
monthly_rev['profit_margin'] = (monthly_rev['profit'] / monthly_rev['revenue'] * 100).round(1)

fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
fig_trend.add_trace(go.Scatter(
    x=monthly_rev['period'], y=monthly_rev['revenue'],
    name='Revenue', fill='tozeroy',
    line=dict(color='#2D4CFF', width=2.5),
    fillcolor='rgba(45,76,255,0.1)'
), secondary_y=False)
fig_trend.add_trace(go.Scatter(
    x=monthly_rev['period'], y=monthly_rev['profit_margin'],
    name='Profit Margin %', mode='lines+markers',
    line=dict(color='#FFD60A', width=2), marker=dict(size=6)
), secondary_y=True)

fig_trend.update_layout(**PLOTLY_TEMPLATE['layout'], height=320,
    title=dict(text='Monthly Revenue & Profit Margin Trend', font=dict(size=14)))
fig_trend.update_yaxes(title_text="Net Revenue (₹)", secondary_y=False,
    tickformat=',.0f', tickprefix='₹')
fig_trend.update_yaxes(title_text="Profit Margin (%)", secondary_y=True)
st.plotly_chart(fig_trend, use_container_width=True)

# ─── Region + Category Row ────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Region & Category Analysis</div>', unsafe_allow_html=True)
col_r, col_c = st.columns(2)

with col_r:
    reg_df = fdf.groupby('region').agg(
        revenue=('net_revenue','sum'),
        orders=('order_id','count'),
        margin=('profit_margin_pct','mean')
    ).reset_index().sort_values('revenue', ascending=True)

    fig_reg = go.Figure(go.Bar(
        x=reg_df['revenue'], y=reg_df['region'],
        orientation='h', marker_color=COLORS[:len(reg_df)],
        text=[fmt_inr(v) for v in reg_df['revenue']],
        textposition='outside'
    ))
    fig_reg.update_layout(**PLOTLY_TEMPLATE['layout'], height=280,
        title=dict(text='Revenue by Region', font=dict(size=13)),
        xaxis=dict(tickformat=',.0f', tickprefix='₹'))
    st.plotly_chart(fig_reg, use_container_width=True)

with col_c:
    cat_df = fdf.groupby('category').agg(
        revenue=('net_revenue','sum')
    ).reset_index().sort_values('revenue', ascending=False)

    fig_cat = px.pie(cat_df, values='revenue', names='category',
                     color_discrete_sequence=COLORS, hole=0.4)
    fig_cat.update_layout(**PLOTLY_TEMPLATE['layout'], height=280,
        title=dict(text='Revenue Share by Category', font=dict(size=13)))
    fig_cat.update_traces(textinfo='label+percent', textfont_size=11)
    st.plotly_chart(fig_cat, use_container_width=True)

# ─── Quarterly Performance ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Quarterly Performance</div>', unsafe_allow_html=True)

q_df = fdf.groupby(['year','quarter']).agg(
    revenue=('net_revenue','sum'),
    profit=('profit','sum'),
    orders=('order_id','count')
).reset_index()
q_df['label']  = q_df['year'].astype(str) + '-' + q_df['quarter']
q_df['margin'] = (q_df['profit'] / q_df['revenue'] * 100).round(1)

fig_q = make_subplots(specs=[[{"secondary_y": True}]])
fig_q.add_trace(go.Bar(
    x=q_df['label'], y=q_df['revenue'],
    name='Revenue', marker_color='#2D4CFF', opacity=0.85,
    text=[fmt_inr(v) for v in q_df['revenue']],
    textposition='outside', textfont=dict(size=9)
), secondary_y=False)
fig_q.add_trace(go.Scatter(
    x=q_df['label'], y=q_df['margin'],
    name='Profit Margin %', mode='lines+markers',
    line=dict(color='#FFD60A', width=2.5), marker=dict(size=8, symbol='diamond')
), secondary_y=True)
fig_q.update_layout(**PLOTLY_TEMPLATE['layout'], height=320,
    title=dict(text='Quarterly Revenue vs Profit Margin', font=dict(size=13)))
fig_q.update_yaxes(title_text="Revenue (₹)", secondary_y=False, tickformat=',.0f', tickprefix='₹')
fig_q.update_yaxes(title_text="Profit Margin (%)", secondary_y=True)
st.plotly_chart(fig_q, use_container_width=True)

# ─── Top Products + Sales Rep ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Top Performers</div>', unsafe_allow_html=True)
col_p, col_s = st.columns(2)

with col_p:
    top_prod = fdf.groupby('product')['net_revenue'].sum().sort_values(ascending=False).head(10).reset_index()
    top_prod.columns = ['product', 'revenue']

    fig_prod = go.Figure(go.Bar(
        x=top_prod['revenue'][::-1], y=top_prod['product'][::-1],
        orientation='h', marker_color='#2D4CFF',
        text=[fmt_inr(v) for v in top_prod['revenue'][::-1]],
        textposition='outside'
    ))
    fig_prod.update_layout(**PLOTLY_TEMPLATE['layout'], height=360,
        title=dict(text='Top 10 Products by Revenue', font=dict(size=13)),
        xaxis=dict(tickformat=',.0f', tickprefix='₹'))
    st.plotly_chart(fig_prod, use_container_width=True)

with col_s:
    rep_df = fdf.groupby('sales_rep').agg(
        revenue=('net_revenue','sum'),
        orders=('order_id','count'),
        margin=('profit_margin_pct','mean')
    ).reset_index().sort_values('revenue', ascending=False)

    fig_rep = go.Figure()
    fig_rep.add_trace(go.Bar(
        x=rep_df['revenue'], y=rep_df['sales_rep'],
        orientation='h',
        marker=dict(
            color=rep_df['revenue'],
            colorscale=[[0,'#1a2866'],[0.5,'#2D4CFF'],[1,'#6B8FFF']],
            showscale=False
        ),
        text=[fmt_inr(v) for v in rep_df['revenue']],
        textposition='outside'
    ))
    fig_rep.update_layout(**PLOTLY_TEMPLATE['layout'], height=360,
        title=dict(text='Sales Rep Revenue Performance', font=dict(size=13)),
        yaxis=dict(categoryorder='total ascending'),
        xaxis=dict(tickformat=',.0f', tickprefix='₹'))
    st.plotly_chart(fig_rep, use_container_width=True)

# ─── Heatmap ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔥 Category × Month Revenue Heatmap</div>', unsafe_allow_html=True)

month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
heat = fdf.pivot_table(values='net_revenue', index='category', columns='month_num', aggfunc='sum').fillna(0)
heat.columns = [month_names[c-1] for c in heat.columns]

fig_heat = px.imshow(heat/1e5,
    labels=dict(x="Month", y="Category", color="Revenue (₹L)"),
    color_continuous_scale='Blues', text_auto='.0f', aspect='auto')
fig_heat.update_layout(**PLOTLY_TEMPLATE['layout'], height=300,
    title=dict(text='Revenue Heatmap by Category & Month (₹ Lakhs)', font=dict(size=13)))
st.plotly_chart(fig_heat, use_container_width=True)

# ─── Channel + Payment ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💳 Channel & Payment Analysis</div>', unsafe_allow_html=True)
col_ch, col_pay = st.columns(2)

with col_ch:
    ch_df = fdf.groupby('channel')['net_revenue'].sum().reset_index().sort_values('net_revenue', ascending=False)
    fig_ch = px.bar(ch_df, x='channel', y='net_revenue', color='channel',
                    color_discrete_sequence=COLORS,
                    text=[fmt_inr(v) for v in ch_df['net_revenue']])
    fig_ch.update_traces(textposition='outside', showlegend=False)
    fig_ch.update_layout(**PLOTLY_TEMPLATE['layout'], height=280,
        title=dict(text='Revenue by Sales Channel', font=dict(size=13)),
        yaxis=dict(tickformat=',.0f', tickprefix='₹'))
    st.plotly_chart(fig_ch, use_container_width=True)

with col_pay:
    pay_df = fdf.groupby('payment_method')['net_revenue'].sum().reset_index()
    fig_pay = px.pie(pay_df, values='net_revenue', names='payment_method',
                     color_discrete_sequence=COLORS, hole=0.35)
    fig_pay.update_layout(**PLOTLY_TEMPLATE['layout'], height=280,
        title=dict(text='Revenue by Payment Method', font=dict(size=13)))
    fig_pay.update_traces(textinfo='label+percent')
    st.plotly_chart(fig_pay, use_container_width=True)

# ─── ML Forecast Section ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🤖 ML Revenue Forecast (XGBoost)</div>', unsafe_allow_html=True)

try:
    forecast_df = pd.read_csv('reports/6month_forecast.csv', parse_dates=['period'])

    monthly_hist = df.groupby(['year','month_num'])['net_revenue'].sum().reset_index()
    monthly_hist['period'] = pd.to_datetime(
        monthly_hist[['year','month_num']].rename(columns={'year':'year','month_num':'month'}).assign(day=1))
    monthly_hist = monthly_hist.sort_values('period').tail(18)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=monthly_hist['period'], y=monthly_hist['net_revenue'],
        name='Actual Revenue', line=dict(color='#2D4CFF', width=2.5),
        fill='tozeroy', fillcolor='rgba(45,76,255,0.08)'
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df['period'], y=forecast_df['forecast_revenue'],
        name='Forecast', line=dict(color='#1DB954', width=2.5),
        mode='lines+markers', marker=dict(size=8)
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast_df['period'], forecast_df['period'][::-1]]),
        y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
        fill='toself', fillcolor='rgba(29,185,84,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band (±10%)'
    ))
    fig_fc.update_layout(**PLOTLY_TEMPLATE['layout'], height=340,
        title=dict(text='6-Month Revenue Forecast | XGBoost Model (R² = 0.70)', font=dict(size=13)),
        yaxis=dict(tickformat=',.0f', tickprefix='₹'))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("**Forecast Summary Table**")
    fc_display = forecast_df.copy()
    fc_display['Month']            = fc_display['period'].dt.strftime('%B %Y')
    fc_display['Forecast Revenue'] = fc_display['forecast_revenue'].apply(fmt_inr)
    fc_display['Lower Bound']      = fc_display['lower_bound'].apply(fmt_inr)
    fc_display['Upper Bound']      = fc_display['upper_bound'].apply(fmt_inr)
    st.dataframe(fc_display[['Month','Forecast Revenue','Lower Bound','Upper Bound']],
                 use_container_width=True, hide_index=True)
except Exception as e:
    st.info("Run `python src/ml_forecasting.py` first to generate forecast data.")
    st.caption(str(e))

# ─── Raw Data Explorer ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🗃️ Raw Data Explorer"):
    st.markdown(f"**{len(fdf):,} records** after filters")
    cols_show = ['order_id','date','region','category','product','sales_rep','channel',
                 'quantity','net_revenue','profit','profit_margin_pct','discount_pct']
    st.dataframe(fdf[cols_show].sort_values('date', ascending=False).head(500),
                 use_container_width=True, hide_index=True)
    csv = fdf.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data as CSV", csv, "filtered_sales_data.csv", "text/csv")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5060; font-size:0.8rem; padding: 1rem 0;'>"
    "Sales Revenue Intelligence Dashboard · Built with Python, Streamlit, XGBoost & Plotly"
    "</div>", unsafe_allow_html=True
)
