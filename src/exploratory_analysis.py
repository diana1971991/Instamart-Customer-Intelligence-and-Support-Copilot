import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.path.join("data", "instamart.db")
conn = sqlite3.connect(DB_PATH)

# Load clean tables
orders_df = pd.read_sql_query("SELECT * FROM orders", conn)
customers_df = pd.read_sql_query("SELECT * FROM customers", conn)
products_df = pd.read_sql_query("SELECT * FROM products", conn)
conn.close()

# Filter for completed orders
completed_orders = orders_df[orders_df['status'] == 'completed'].copy()
merged_df = completed_orders.merge(customers_df, on='customer_id').merge(products_df, on='product_id')

# (a) Pivot table & Named Aggregations
pivot_tbl = pd.pivot_table (
    merged_df,
    values='order_amount',
    index='city',
    columns='category',
    aggfunc='sum',
    margins=True
)
print(f"\nPivot Table Grand Total: INR {pivot_tbl.loc['All', 'All']:,.2f}")

pivot_result = pd.pivot_table(
    merged_df,
    index='city',
    values='order_amount',
    aggfunc=['sum', 'mean', 'count'],
    margins=True,          # This adds the grand total rows and columns!
    margins_name='Total'   # Names the summary label 'Total'
).round(2)

print("\nPivot Table with Margins (Grand Totals):")
print(pivot_result)

# (b) Distribution Statistics & Outliers for completed order_amount
amt = completed_orders['order_amount']
mean_val = amt.mean()
median_val = amt.median()
skew_val = amt.skew()
q1 = amt.quantile(0.25)
q3 = amt.quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
lower_fence = q1 - 1.5 * iqr

outliers_above = (amt > upper_fence).sum()
outliers_below = (amt < lower_fence).sum()

print("\n------- Distribution & Outliers for completed order_amount -------")
print(f"Mean: {mean_val:.2f} | Median: {median_val:.2f} | Skewness: {skew_val:.2f}")
print(f"Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
print(f"Upper Fence: {upper_fence:.2f} | Outliers Above: {outliers_above}")
print(f"Lower Fence: {lower_fence:.2f} | Outliers Below: {outliers_below}")

# Correlation Matrix
corr_cols = ['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']
corr_matrix = completed_orders[corr_cols].corr()
print("\n------- Correlation Matrix -------")
print(corr_matrix)

# --- Answers to the Exploratory Questions ---
exploratory_answers = """
=== Answers to Exploratory Questions ===

1. What is this data about?
The dataset contains e-commerce grocery delivery transactions from Instamart across multiple Indian cities, capturing order amounts, quantities, discount percentages, fulfillment statuses, and delivery times.

2. What do its distributions and statistics look like?
Order amounts are heavily right-skewed(mean ~269.91 vs median 135.00, skewness ~4.61), reflecting a long tail of high-value basket orders.

3. Are there outliers or anomalies?
Yes. Using 1.5×IQR fences (upper fence 645.30), there are 897 outliers above the upper fence and none below, representing unusually large bulk or premium purchases.

4. What relationships exist between variables?
The largest correlation in absolute value among the tested features is between order_amount and discount_pct at about -0.095 (while order_amount and quantity is +0.087, slightly smaller in absolute value). Note that correlation of this size does not establish causation.
"""
print(exploratory_answers)