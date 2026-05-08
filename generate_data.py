"""
Sales & Revenue Intelligence Dashboard
Data Generation Script
Generates realistic 3-year sales dataset for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ─── Constants ────────────────────────────────────────────────────────────────
REGIONS = ['North', 'South', 'East', 'West', 'Central']
CATEGORIES = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports', 'Books', 'Beauty']
PRODUCTS = {
    'Electronics': ['Laptop Pro X1', 'Wireless Headphones', 'Smart Watch', 'Tablet Ultra', '4K Monitor', 'USB-C Hub'],
    'Clothing':    ['Denim Jacket', 'Running Shoes', 'Formal Shirt', 'Yoga Pants', 'Winter Coat', 'Sneakers'],
    'Home & Kitchen': ['Air Fryer', 'Coffee Maker', 'Blender Pro', 'Induction Cooktop', 'Knife Set', 'Food Processor'],
    'Sports':      ['Yoga Mat', 'Resistance Bands', 'Dumbbell Set', 'Cycling Helmet', 'Sports Bag', 'Jump Rope'],
    'Books':       ['Python for Data Science', 'Business Analytics', 'Leadership Mastery', 'ML Handbook', 'Finance 101', 'AI Revolution'],
    'Beauty':      ['Face Serum', 'Moisturizer SPF50', 'Hair Mask', 'Vitamin C Cream', 'Eye Cream', 'Lip Balm Set'],
}
PRICE_RANGES = {
    'Electronics': (5000, 80000),
    'Clothing':    (500,  8000),
    'Home & Kitchen': (1000, 20000),
    'Sports':      (300,  10000),
    'Books':       (200,  2000),
    'Beauty':      (200,  5000),
}
COST_MARGIN = {
    'Electronics': 0.60,
    'Clothing':    0.45,
    'Home & Kitchen': 0.50,
    'Sports':      0.40,
    'Books':       0.30,
    'Beauty':      0.35,
}
SALES_REPS = [
    'Anjali Sharma', 'Rohit Verma', 'Priya Nair', 'Suresh Kumar',
    'Deepika Menon', 'Arjun Patel', 'Kavya Reddy', 'Vikram Singh',
    'Sneha Joshi', 'Manish Gupta'
]
CHANNELS = ['Online', 'Retail Store', 'Distributor', 'Direct Sales']
PAYMENT_METHODS = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash']

def seasonal_factor(month):
    """Returns seasonal demand multiplier"""
    factors = {1: 0.75, 2: 0.72, 3: 0.85, 4: 0.90, 5: 0.92,
               6: 0.88, 7: 0.80, 8: 0.85, 9: 0.95, 10: 1.10,
               11: 1.40, 12: 1.50}
    return factors.get(month, 1.0)

def growth_factor(date, base_date):
    """Returns YoY growth multiplier"""
    months_elapsed = (date.year - base_date.year) * 12 + (date.month - base_date.month)
    return 1 + (0.015 * months_elapsed)  # ~18% annual growth

def generate_sales_data(n_records=5000):
    records = []
    start_date = datetime(2022, 1, 1)
    end_date   = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    for i in range(n_records):
        date = start_date + timedelta(days=random.randint(0, date_range))
        region = random.choice(REGIONS)
        category = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS[category])
        price_min, price_max = PRICE_RANGES[category]

        base_price = round(random.uniform(price_min, price_max), -1)
        quantity   = random.randint(1, 20)

        # Apply seasonal + growth factors
        sfactor = seasonal_factor(date.month)
        gfactor = growth_factor(date, start_date)
        quantity = max(1, int(quantity * sfactor * gfactor * random.uniform(0.8, 1.2)))

        revenue    = round(base_price * quantity, 2)
        cost       = round(revenue * COST_MARGIN[category], 2)
        profit     = round(revenue - cost, 2)
        profit_pct = round((profit / revenue) * 100, 2)

        discount   = random.choice([0, 0, 0, 5, 10, 15, 20])
        disc_amt   = round(revenue * discount / 100, 2)
        net_rev    = round(revenue - disc_amt, 2)

        region_factor = {'North': 1.1, 'South': 1.05, 'East': 0.95, 'West': 1.15, 'Central': 1.0}
        net_rev = round(net_rev * region_factor.get(region, 1.0), 2)

        records.append({
            'order_id':        f'ORD-{10000 + i}',
            'date':            date.strftime('%Y-%m-%d'),
            'month':           date.strftime('%B'),
            'month_num':       date.month,
            'quarter':         f'Q{((date.month - 1) // 3) + 1}',
            'year':            date.year,
            'region':          region,
            'category':        category,
            'product':         product,
            'sales_rep':       random.choice(SALES_REPS),
            'channel':         random.choice(CHANNELS),
            'payment_method':  random.choice(PAYMENT_METHODS),
            'unit_price':      base_price,
            'quantity':        quantity,
            'gross_revenue':   revenue,
            'discount_pct':    discount,
            'discount_amount': disc_amt,
            'net_revenue':     net_rev,
            'cost':            cost,
            'profit':          profit,
            'profit_margin_pct': profit_pct,
            'customer_id':     f'CUST-{random.randint(1000, 5000)}',
            'return_flag':     random.choices([0, 1], weights=[92, 8])[0],
        })

    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    return df

def generate_targets(df):
    """Generate monthly sales targets (slightly above actual for realism)"""
    monthly_actual = df.groupby(['year', 'month_num'])['net_revenue'].sum().reset_index()
    monthly_actual.columns = ['year', 'month_num', 'actual_revenue']
    monthly_actual['target_revenue'] = (monthly_actual['actual_revenue'] * random.uniform(1.05, 1.15)).round(2)
    monthly_actual['achievement_pct'] = ((monthly_actual['actual_revenue'] / monthly_actual['target_revenue']) * 100).round(2)
    return monthly_actual

if __name__ == '__main__':
    print("Generating sales data...")
    df = generate_sales_data(5000)
    df.to_csv('sales_data.csv', index=False)
    print(f"  ✓ sales_data.csv  — {len(df)} records")

    targets = generate_targets(df)
    targets.to_csv('monthly_targets.csv', index=False)
    print(f"  ✓ monthly_targets.csv — {len(targets)} records")

    # Summary stats
    print(f"\n  Total Revenue : ₹{df['net_revenue'].sum():,.0f}")
    print(f"  Total Orders  : {len(df):,}")
    print(f"  Avg Order Val : ₹{df['net_revenue'].mean():,.0f}")
    print(f"  Profit Margin : {df['profit_margin_pct'].mean():.1f}%")
    print(f"  Date Range    : {df['date'].min()} → {df['date'].max()}")
