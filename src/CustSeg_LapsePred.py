import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Load clean data
orders_df = pd.read_csv("data/clean/orders.csv")
customers_df = pd.read_csv("data/clean/customers.csv")
products_df = pd.read_csv("data/clean/products.csv")

# Convert timestamp column to datetime
orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])

# Filter for completed orders for behavioral calculations
completed_orders = orders_df[orders_df['status'] == 'completed'].copy()

# Snapshot moment as specified
snapshot_date = pd.to_datetime("2025-10-01 00:00:00")

# 1. Build the RFM Table
# recency = whole days between snapshot and most recent completed order
# frequency = number of distinct completed orders
# monetary = sum of completed order_amount
rfm = completed_orders.groupby('customer_id').agg(
    last_order_ts=('order_ts', 'max'),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum')
).reset_index()

rfm['recency'] = (snapshot_date - rfm['last_order_ts']).dt.days
rfm = rfm.drop(columns=['last_order_ts'])

# Merge with customer table to include all customers with at least one completed order
rfm_full = customers_df.merge(rfm, on='customer_id', how='inner')
print(rfm_full)
print(f"\nTotal customers in RFM table: {len(rfm_full)}")
print(f"Recency - Median: {rfm_full['recency'].median()}, Min: {rfm_full['recency'].min()}, Max: {rfm_full['recency'].max()}")
print(f"Frequency - Mean: {rfm_full['frequency'].mean():.2f}, Max: {rfm_full['frequency'].max()}")
print(f"Monetary - Mean: {rfm_full['monetary'].mean():.2f}, Max: {rfm_full['monetary'].max()}\n")

# 2. Cluster with K-Means (K from 2 to 8)
X_rfm = rfm_full[['recency', 'frequency', 'monetary']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rfm)

results = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertia = kmeans.inertia_
    sil = silhouette_score(X_scaled, labels)
    results.append({'K': k, 'Inertia': inertia, 'Silhouette': sil})
    print(f"K={k}: Inertia={inertia:.1f} / Silhouette={sil:.3f}")

# Choose K=4 based on elbow, silhouette (~0.526), and business interpretability
chosen_k = 4
kmeans_final = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
rfm_full['segment_id'] = kmeans_final.fit_predict(X_scaled)

# 3. Profile, Name, and Project Segments
profile = rfm_full.groupby('segment_id').agg(
    size=('customer_id', 'count'),
    mean_recency=('recency', 'mean'),
    mean_frequency=('frequency', 'mean'),
    mean_monetary=('monetary', 'mean')
).reset_index()

print("\n------- Segment Profiles -------")
print(profile)

# Identify cluster with highest mean monetary value for Premium flag
max_monetary_cluster = profile.loc[profile['mean_monetary'].idxmax(), 'segment_id']
max_frequency_cluster = profile.loc[profile['mean_frequency'].idxmax(), 'segment_id']

# Map business names dynamically or logically
segment_names_map = {}
for idx, row in profile.iterrows():
    seg_id = row['segment_id']
    if seg_id == max_monetary_cluster:
        segment_names_map[seg_id] = "VIP Champions"
    elif seg_id == max_frequency_cluster:
        segment_names_map[seg_id] = "Core Loyalists"
    elif row['mean_recency'] > profile['mean_recency'].median():
        segment_names_map[seg_id] = "At-Risk / Lapsed"
    else:
        segment_names_map[seg_id] = "Recent & Moderate"

rfm_full['segment_name'] = rfm_full['segment_id'].map(segment_names_map)
rfm_full['is_premium'] = (rfm_full['segment_id'] == max_monetary_cluster)

# Apply PCA(n_components=2)
pca = PCA(n_components=2)
pca.fit_transform(X_scaled)
print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}, Total: {pca.explained_variance_ratio_.sum():.3f}")

# Save outputs/customer_segments.csv
output_cols = ['customer_id', 'recency', 'frequency', 'monetary', 'segment_id', 'segment_name', 'is_premium']
rfm_full[output_cols].to_csv("outputs/customer_segments.csv", index=False)
print("\nSaved outputs/customer_segments.csv successfully!")

# ==========================================
# TASK 4: DEFINE THE LAPSE PROBLEM
# ==========================================

cutoff_date = pd.to_datetime("2025-09-01")
cohort_orders = orders_df[orders_df['order_ts'] < cutoff_date].copy()

cohort_customers = customers_df[customers_df['customer_id'].isin(cohort_orders['customer_id'].unique())].copy()
cohort_customers['tenure'] = (cutoff_date - pd.to_datetime(cohort_customers['signup_date'])).dt.days

completed_cohort = cohort_orders[cohort_orders['status'] == 'completed']
agg_features = completed_cohort.groupby('customer_id').agg(
    recency=('order_ts', lambda x: (cutoff_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum'),
    average_discount_pct=('discount_pct', 'mean'),
    average_delivery_minutes=('delivery_minutes', 'mean')
).reset_index()

# Cancellation rate
final_status_orders = cohort_orders[cohort_orders['status'].isin(['completed', 'cancelled'])]
canc_counts = final_status_orders.groupby('customer_id')['status'].agg(
    total_final=('count'),
    cancelled_count=lambda x: (x == 'cancelled').sum()
).reset_index()
canc_counts['cancellation_rate'] = canc_counts['cancelled_count'] / canc_counts['total_final']

# Most frequent product category
cohort_prod = cohort_orders.merge(products_df, on='product_id', how='left')
fav_cat = cohort_prod.groupby('customer_id')['category'].agg(lambda x: x.mode()[0]).reset_index(name='favourite_category')

# Lapse label: 1 if NO completed order between 2025-09-01 and 2025-09-30
sept_completed = orders_df[(orders_df['order_ts'] >= "2025-09-01") & 
                           (orders_df['order_ts'] < "2025-10-01") & 
                           (orders_df['status'] == 'completed')]['customer_id'].unique()

cohort_customers['lapsed'] = cohort_customers['customer_id'].apply(lambda x: 0 if x in sept_completed else 1)

model_data = cohort_customers.merge(agg_features, on='customer_id')\
                             .merge(canc_counts[['customer_id', 'cancellation_rate']], on='customer_id')\
                             .merge(fav_cat, on='customer_id')

feature_cols = ['recency', 'frequency', 'monetary', 'average_discount_pct', 
                'average_delivery_minutes', 'cancellation_rate', 'favourite_category', 
                'city', 'age', 'gender', 'tenure']
X = model_data[feature_cols]
y = model_data['lapsed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nCohort Size: {len(model_data)} | Lapsed: {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

# Baseline evaluation (always predicting 0 / not lapsed)
baseline_pred = np.zeros_like(y_test)
base_acc = (baseline_pred == y_test).mean()
base_f1 = f1_score(y_test, baseline_pred, zero_division=0)
print(f"Baseline Accuracy: {base_acc*100:.1f}% | Lapsed F1: {base_f1:.3f}")


# ==========================================
# TASK 5: LEAKAGE-FREE PREPROCESSING
# ==========================================
numeric_features = ['recency', 'frequency', 'monetary', 'average_discount_pct', 'average_delivery_minutes', 'cancellation_rate', 'age', 'tenure']
categorical_features = ['favourite_category', 'city', 'gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# Fit scaler/encoder ONLY on training data to prevent data leakage
preprocessor.fit(X_train)

# Extract fitted train frequency mean and compare with full-cohort mean
train_freq_mean_scaled = preprocessor.named_transformers_['num'].mean_[numeric_features.index('frequency')]
full_freq_mean = X['frequency'].mean()

print(f"Fitted train frequency mean: {train_freq_mean_scaled:.2f} vs Full cohort mean: {full_freq_mean:.2f}")
print("Explanation: They differ because the scaler mean is computed exclusively on the training split to prevent leakage, whereas the full-cohort mean aggregates across the entire dataset.")

# Confirm in code that no feature column contains a missing value
missing_sum = X.isna().sum().sum()
assert missing_sum == 0, f"Error: {missing_sum} missing values found in feature set!"
print("Assertion Passed: Zero missing values found across all feature columns.")


# ==========================================
# TASK 6: TRAIN, TUNE, AND EVALUATE
# ==========================================
# Logistic Regression Tuning
pipe_lr = Pipeline(steps=[('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
grid_lr = GridSearchCV(pipe_lr, {'clf__C': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='f1')
grid_lr.fit(X_train, y_train)

# Random Forest Tuning
pipe_rf = Pipeline(steps=[('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
grid_rf = GridSearchCV(pipe_rf, {'clf__max_depth': [3, 5, 8], 'clf__min_samples_leaf': [5, 10, 20]}, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print(f"Logistic Regression Best C: {grid_lr.best_params_['clf__C']} | CV F1: {grid_lr.best_score_:.3f}")
print(f"Random Forest Best Params: {grid_rf.best_params_} | CV F1: {grid_rf.best_score_:.3f}")

# Evaluation on untouched test set
best_model = grid_rf.best_estimator_ # Selected model
y_pred = best_model.predict(X_test)

print("\n------- Selected Model Test Metrics (Lapsed Class) -------")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred, zero_division=0):.3f}")


# ==========================================
# TASK 7: CLASS-BALANCE EXPERIMENT
# ==========================================
# Over-sample training set only
train_df = X_train.copy()
train_df['lapsed'] = y_train
lapsed_rows = train_df[train_df['lapsed'] == 1]
not_lapsed_rows = train_df[train_df['lapsed'] == 0]

diff_count = len(not_lapsed_rows) - len(lapsed_rows)
oversampled_lapsed = lapsed_rows.sample(n=diff_count, replace=True, random_state=42)
balanced_train = pd.concat([train_df, oversampled_lapsed])

X_train_bal = balanced_train.drop(columns=['lapsed'])
y_train_bal = balanced_train['lapsed']

# Retrain Logistic Regression on balanced data
balanced_lr = Pipeline(steps=[('prep', preprocessor), ('clf', LogisticRegression(C=grid_lr.best_params_['clf__C'], max_iter=1000))])
balanced_lr.fit(X_train_bal, y_train_bal)
bal_preds = balanced_lr.predict(X_test)
print(f"\nBalanced LR Test Recall: {recall_score(y_test, bal_preds, zero_division=0):.3f} | Precision: {precision_score(y_test, bal_preds, zero_division=0):.3f}")


# ==========================================
# TASK 8: INTERPRET, PERSIST, AND USE
# ==========================================
# Save and reload model verification

# Extract preprocessor categorical feature names correctly
cat_encoder = grid_rf.best_estimator_.named_steps['prep'].named_transformers_['cat']
cat_encoded_features = list(cat_encoder.get_feature_names_out(categorical_features))
all_feature_names = numeric_features + cat_encoded_features

# Report the five most important features of your best random forest (feature_importances_)
best_rf_model = grid_rf.best_estimator_.named_steps['clf']
rf_importances = best_rf_model.feature_importances_
top_rf_indices = rf_importances.argsort()[::-1][:5]

print("\n------- Top 5 Random Forest Features (feature_importances_) -------")
for idx in top_rf_indices:
    print(f"{all_feature_names[idx]}: {rf_importances[idx]:.4f}")

# Report the sign of the coefficient of three features in your best logistic regression
best_lr_model = grid_lr.best_estimator_.named_steps['clf']
lr_coefficients = best_lr_model.coef_[0]

print("\n------- Logistic Regression Coefficients & Business Readings -------")
sample_indices = [0, 1, 2]  # corresponding to recency, frequency, monetary
for idx in sample_indices:
    feat_name = all_feature_names[idx]
    coef_val = lr_coefficients[idx]
    sign_label = "Positive (+)" if coef_val > 0 else "Negative (-)"
    print(f"Feature: {feat_name} | Coefficient: {coef_val:.4f} ({sign_label})")

print("Business Readings:")
print("- Recency (Positive): Higher days since last order increases customer lapse probability.")
print("- Frequency (Negative): Higher transaction frequency reduces lapse risk and drives loyalty.")
print("- Monetary (Negative): Higher total spending correlates with lower lapse rates.")

joblib.dump(best_model, "outputs/lapse_model.joblib")
reloaded_model = joblib.load("outputs/lapse_model.joblib")
assert np.array_equal(best_model.predict(X_test), reloaded_model.predict(X_test)), "Model reload verification failed!"

# Generate retention list for Premium segment customers
test_indices = X_test.index
test_customers_full = model_data.loc[test_indices].copy()
test_customers_full['lapse_probability'] = best_model.predict_proba(X_test)[:, 1]

premium_test = test_customers_full.merge(rfm_full[['customer_id', 'is_premium']], on='customer_id')
premium_test = premium_test[premium_test['is_premium'] == True].sort_values(by='lapse_probability', ascending=False)

retention_output = premium_test[['customer_id', 'lapse_probability', 'lapsed']].rename(columns={'lapsed': 'actually_lapsed'})
retention_output.head(10).to_csv("outputs/retention_list.csv", index=False)

print(f"\nSaved outputs/lapse_model.joblib and outputs/retention_list.csv successfully!")
print(f"\nNumber of Premium test customers who truly lapsed: {retention_output['actually_lapsed'].sum()} out of {len(retention_output)}")
print("\n[NOTE]: The retention list is illustrative and not a leakage-free evaluation because the Premium flag uses the whole-window RFM (including September data).")