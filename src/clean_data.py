import os
import sqlite3
import pandas as pd
import numpy as np

# PART A: DATA FOUNDATION - AUDIT, CLEAN, & STORE

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)
DB_PATH = os.path.join("data", "instamart.db")

# ==========================================================
# 1. LOADING RAW TABLES
# ==========================================================

cust_df = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
prod_df = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
orders_df = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
tickets_df = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False)

tables = {
    "customers.csv": cust_df,
    "products.csv": prod_df,
    "orders.csv": orders_df,
    "tickets.csv": tickets_df
}

# ==========================================================
# 2. AUDIT RAW DATA (Shape, Dtypes, Missing Count & Percentage)
# ==========================================================

print("\n------- TABLE SHAPES, DATATYPES & MISSING VALUES -------")
for name, df in tables.items():
    print(f"\nTable: {name}")
    print(f"Shape: {df.shape}")
    
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    
    audit_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Count": missing_count,
        "Missing Percentage (%)": missing_pct
    })
    print(audit_df)
print("\n============================================================")

print("\n------- SPECIFIC AUDIT FIGURES -------")
raw_orders_rows = len(orders_df)
exact_duplicates = orders_df.duplicated().sum()
pending_rows = (orders_df['status'] == 'pending').sum()
distinct_order_ids = orders_df['order_id'].nunique()

# Non-numeric amounts
def check_numeric_amount(val):
    if pd.isna(val) or str(val).strip() == "":
        return True
    try:
        float(val)
        return False
    except (ValueError, TypeError):
        return True

non_numeric_amounts = orders_df['order_amount'].apply(check_numeric_amount).sum()

# Delivery times checks
deliv_numeric = pd.to_numeric(orders_df['delivery_minutes'], errors='coerce')
neg_delivery = (deliv_numeric < 0).sum()
blank_delivery = (orders_df['delivery_minutes'] == "").sum() + orders_df['delivery_minutes'].isna().sum()


# Customers checks
cust_age_numeric = pd.to_numeric(cust_df['age'], errors='coerce')
blank_ages = cust_df['age'].isna().sum() + (cust_df['age'] == "").sum()
out_of_bounds_ages = ((cust_age_numeric < 10) | (cust_age_numeric > 100)).sum()
distinct_cities = cust_df['city'].nunique()
non_blank_genders = cust_df['gender'].dropna()
distinct_genders = non_blank_genders[non_blank_genders != ""].nunique()
blank_genders = cust_df['gender'].isna().sum() + (cust_df['gender'] == "").sum()

print(f"raw orders.csv rows: {raw_orders_rows}")
print(f"Number of exact duplicate rows: {exact_duplicates}")
print(f"Number of rows whose status is pending: {pending_rows}")
print(f"Number of distinct order_id values: {distinct_order_ids}")
print(f"Number of non-numeric order_amount entries: {non_numeric_amounts}")
print(f"Number of negative delivery_minutes: {neg_delivery}")
print(f"Number of blank delivery_minutes: {blank_delivery}")
print(f"Number of blank age: {blank_ages}")
print(f"Number of age values outside 10-100: {out_of_bounds_ages}")
print(f"Number of distinct raw city spellings: {distinct_cities}")
print(f"Number of distinct non-blank raw gender spellings: {distinct_genders}")
print(f"Number of blank gender: {blank_genders}")
print("\n============================================================")
# ==========================================================
# 3. CLEAN CUSTOMERS TABLE
# ========================================================== 

print("\n------- CLEANING CUSTOMERS -------")
cust_df['city'] = cust_df['city'].str.strip().str.title()

def map_gender(g):
    if pd.isna(g) or str(g).strip() == "":
        return "unknown"
    g_lower = str(g).strip().lower()
    if g_lower in ['m', 'male']:
        return 'male'
    elif g_lower in ['f', 'female']:
        return 'female'
    return 'unknown'

cust_df['gender'] = cust_df['gender'].apply(map_gender)

cust_df['age'] = pd.to_numeric(cust_df['age'], errors='coerce')
cust_df.loc[(cust_df['age'] < 10) | (cust_df['age'] > 100), 'age'] = np.nan
city_medians = cust_df.groupby('city')['age'].transform('median')
cust_df['age'] = cust_df['age'].fillna(city_medians)

cust_df['signup_date'] = pd.to_datetime(cust_df['signup_date'])

city_counts = cust_df['city'].value_counts().to_dict()
gender_counts = cust_df['gender'].value_counts().to_dict()
missing_ages = cust_df['age'].isna().sum()
min_age = int(cust_df['age'].min())
max_age = int(cust_df['age'].max())

print("Customers cleaning completed.")
print(f"  - Cities: {', '.join([f'{c} {count}' for c, count in city_counts.items()])}")
print(f"  - Genders: {', '.join([f'{g} {count}' for g, count in gender_counts.items()])}")
print(f"  - Missing ages: {missing_ages} | Age range: {min_age} to {max_age}")
print("\n============================================================")

# ==========================================================
# 4. CLEAN ORDERS TABLE
# ==========================================================
print("\n------- CLEANING ORDERS -------")
orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])
orders_df = orders_df.drop_duplicates()
orders_df = orders_df.sort_values('order_ts').drop_duplicates(subset=['order_id'], keep='last')

prod_prices = prod_df.set_index('product_id')['price_inr'].to_dict()

def parse_amt(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

orders_df['order_amount_num'] = orders_df['order_amount'].apply(parse_amt)

def fallback_amt(row):
    price = prod_prices.get(row['product_id'], 0)
    return round(price * row['quantity'] * (1 - row['discount_pct'] / 100), 2)

missing_amt = orders_df['order_amount_num'].isna()
orders_df.loc[missing_amt, 'order_amount_num'] = orders_df[missing_amt].apply(fallback_amt, axis=1)
orders_df['order_amount'] = orders_df['order_amount_num']
orders_df = orders_df.drop(columns=['order_amount_num'])

# Handle negative or missing delivery_minutes cleanly
orders_df['delivery_minutes'] = pd.to_numeric(orders_df['delivery_minutes'], errors='coerce')
orders_df.loc[orders_df['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan

# Impute delivery time for completed orders using product_id median, then global completed median, then default 25.0
orders_df['delivery_minutes'] = orders_df['delivery_minutes'].fillna(
    orders_df[orders_df['status'] == 'completed'].groupby('product_id')['delivery_minutes'].transform('median')
)
global_median = orders_df[orders_df['status'] == 'completed']['delivery_minutes'].median()
if pd.isna(global_median):
    global_median = 25.0
orders_df['delivery_minutes'] = orders_df['delivery_minutes'].fillna(global_median)

# Cancelled orders should have missing delivery times (never delivered)
orders_df.loc[orders_df['status'] == 'cancelled', 'delivery_minutes'] = np.nan

# Format order_ts back to string YYYY-MM-DD HH:MM:SS for SQLite
orders_df['order_ts'] = orders_df['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

total_clean_orders = len(orders_df)
status_counts = orders_df['status'].value_counts().to_dict()

pending_remaining = (orders_df['status'] == 'pending').sum()
missing_amounts = orders_df['order_amount'].isna().sum()
missing_delivery_completed = orders_df[orders_df['status'] == 'completed']['delivery_minutes'].isna().sum()
missing_delivery_cancelled = orders_df[orders_df['status'] == 'cancelled']['delivery_minutes'].isna().sum()
total_completed_amount = orders_df[orders_df['status'] == 'completed']['order_amount'].sum()

print("Orders cleaning completed.")
print(f"  - Total clean orders: {total_clean_orders}")
print(f"  - Status breakdown: {', '.join([f'{status} {count}' for status, count in status_counts.items()])}")
print(f"  - Pending rows remaining: {pending_remaining}")
print(f"  - Missing order_amount entries: {missing_amounts}")
print(f"  - Missing delivery_minutes (completed orders): {missing_delivery_completed}")
print(f"  - Missing delivery_minutes (cancelled orders): {missing_delivery_cancelled}")
print(f"  - Sum of order_amount (completed orders): INR {total_completed_amount:,.2f}")
print("\n============================================================")

# ==========================================================
# 5. STORE AND VALIDATE
# ==========================================================
print("\n------- STORING CLEAN TABLES & VALIDATING -------")
cust_df.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
prod_df.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
orders_df.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)

conn = sqlite3.connect(DB_PATH)
cust_df.to_sql("customers", conn, if_exists="replace", index=False)
prod_df.to_sql("products", conn, if_exists="replace", index=False)
orders_df.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()

# 9 Validation Checkpoints Assertions
assert len(orders_df) == 12309, "Checkpoint 1 failed: Final order count mismatch!"
assert orders_df['order_amount'].isna().sum() == 0, "Checkpoint 2 failed: Missing order amounts found!"
assert len(cust_df[cust_df['age'].isna()]) == 0, "Checkpoint 3 failed: Missing customer ages found!"
assert orders_df[orders_df['status'] == 'completed']['delivery_minutes'].isna().sum() == 0, "Checkpoint 4 failed: Missing delivery times in completed orders!"
assert len(orders_df[orders_df['status'] == 'pending']) == 0, "Checkpoint 5 failed: Pending orders still present!"
assert len(cust_df) == 1200, "Checkpoint 6 failed: Customer count mismatch!"
assert len(prod_df) == 40, "Checkpoint 7 failed: Product count mismatch!"
assert orders_df['delivery_minutes'].dropna().min() >= 0, "Checkpoint 8 failed: Negative delivery minutes found!"
assert cust_df['age'].dropna().between(10, 100).all(), "Checkpoint 9 failed: Customer age out of bounds [10-100]!"


# FINAL DATABASE & CHECKPOINT VALIDATION ---
import sqlite3

conn = sqlite3.connect("data/instamart.db")
cursor = conn.cursor()

# Check tables in SQLite database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
conn.close()

print(f"Tables found in data/instamart.db: {tables}")

# Verify all 3 required tables exist
required_tables = {'customers', 'products', 'orders'}
if required_tables.issubset(set(tables)):
    print("ALL CHECKPOINTS PASSED")
else:
    print("WARNING: Some required tables are missing!")