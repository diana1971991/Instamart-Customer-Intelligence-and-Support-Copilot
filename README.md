# Instamart End-to-End Capstone Project

## Overview
This repository contains the complete end-to-end Capstone project for Instamart, structured into four core modules: Exploratory Data Analysis & SQL (Part A), Machine Learning Customer Lapse Prediction (Part B), Support Ticket Triage with PII Guardrails (Part C), and Policy RAG & Autonomous Refund Agent (Part D).

## Reproduce
Follow these numbered shell commands to set up the environment and execute the complete pipeline from terminal:

1. `python -m venv .venv`
2. `.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `python run.py`

## Part A
* **Data Sources Processed**: `customers.csv`, `products.csv`, `orders.csv`
* **SQLite Database**: `data/instamart.db` with tables `customers`, `products`, `orders`
* **Data Cleaning Rules (Clean Orders - 5 Marks Documentation)**:
  1. **Datetime Conversion**: Converted `order_ts` to a standardized datetime format.
  2. **Duplicate Removal**: Removed exact duplicate rows from the dataset.
  3. **Duplicate Resolution by Timestamp**: For orders appearing multiple times with different statuses (e.g., earlier pending record and later final record), kept only the record with the latest `order_ts` for each `order_id`.
  4. **Amount Imputation**: Converted `order_amount` to numeric, turning unparseable entries into missing values, then filled each missing amount from the product price formula: `order_amount = round(price_inr × quantity × (1 − discount_pct / 100), 2)`.
  5. **Delivery Minutes Imputation**: Set negative `delivery_minutes` to missing. For orders with status `completed`, filled every missing `delivery_minutes` with the median delivery time of that order's product category among completed orders using `groupby` and `transform`. For orders with status `cancelled`, left `delivery_minutes` missing (as cancelled orders were never delivered).
* **Post-Cleaning Reference Check**:
  * Total rows: `12,309` (one per `order_id`)
  * Status breakdown: `11,110` completed and `1,199` cancelled (`0` pending rows)
  * Missing check: `0` missing `order_amount`
  * Delivery minutes missing check: `0` missing among completed orders and `1,199` missing among cancelled orders
  * Financial sum: Sum of `order_amount` over completed orders is **INR 2,998,668.80**.
  
* **SQL Aggregation Summary**:

  |    City   | Total Order| Mean Order | Order Count |
  |           |Amount (INR)|Amount (INR)|             |
  |-----------|------------|------------|-------------|
  | Bengaluru | 523,643.40 |   276.47   |    1,894    |
  | Chennai   | 463,629.15 |   272.72   |    1,700    |
  | Delhi     | 502,829.80 |   243.62   |    2,064    |
  | Hyderabad | 528,589.90 |   284.95   |    1,855    |
  | Mumbai    | 479,058.50 |   297.00   |    1,613    |
  | Pune      | 500,918.05 |   252.48   |    1,984    |

## Part B
* **Classifiers Evaluated**: Logistic Regression vs. Random Forest Classifier
* **Primary Metric (Test F1 Score)**: 
  * Logistic Regression: `0.648` (Best C = 10, passes >0.55 benchmark)
  * Random Forest Classifier: `0.682` (Max Depth = 8, Min Samples Leaf = 20, passes >0.55 benchmark)

### K-Means Clustering
* **Standardization**: Three RFM columns standardized using `StandardScaler`.
* **Clustering Performance Table**:

  | K | Inertia | Silhouette Score |
  |---|---------|------------------|
  | 2 | 1820.8  |       0.465      |
  | 3 | 1268.2  |       0.445      |
  | 4 |  743.1  |       0.526      |
  | 5 |  617.4  |       0.481      |
  | 6 |  503.2  |       0.482      |
  | 7 |  408.5  |       0.416      |
  | 8 |  366.5  |       0.412      |

* **Model Selection (K=4)**: Chosen based on the prominent elbow point at K=4, its peak silhouette score (`0.526`), and strong operational interpretability for targeted marketing campaigns.

* **Inertia Explanation**: Inertia measures the sum of squared distances to cluster centers, which mathematically decreases as more clusters are added because data points sit closer to smaller centers. Therefore, K=8 always yields lower inertia than K=2, but relying on inertia alone would continuously lower until K equals the number of samples. Consequently, silhouette scores and business interpretability must be used alongside the elbow method to select the optimal K.

* **Segment Profiling (K=4)**:
  * *Note: Cluster numbers are arbitrary numeric labels with no intrinsic ordering.*
  * **Segment 1 (Size: ~436)**: Moderate recency/frequency; recommended action: Re-engagement campaigns. (`is_premium: False`)
  * **Segment 2 (Size: ~138, Mean Monetary: ~7,130, Mean Frequency: ~6.8)**: High-value basket buyers; designated as **Premium**. Recommended action: VIP concierge & exclusive perks. (`is_premium: True`)
  * **Segment 3 (Size: ~419, Mean Frequency: ~19.4, Mean Recency: ~8.8 days)**: Core loyalists; recommended action: Loyalty tier rewards. (`is_premium: False`)
  * **Segment 4 (Size: ~182)**: At-risk / low activity; recommended action: Win-back discount incentives. (`is_premium: False`)

* **PCA Dimensionality Reduction**:
  * `explained_variance_ratio_`: `[0.634, 0.218]` (Total explained variance: `0.852`, within tolerance).

* **Generated Outputs**: `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`

* **Data Leakage & Methodology Guarantees**:
  * **No Data Leakage**: All encoders (`OneHotEncoder`), scalers (`StandardScaler`), and resamplers saw training rows exclusively; no transformations or fits touched test data.
  * **Temporal Isolation**: No data from `2025-09-01` onwards entered any predictive feature (ensuring strict historical simulation).
  * **Feature Independence**: Unsupervised cluster labels and post-hoc flags from Task 3 (such as `segment_id` and `is_premium`) were strictly excluded from model feature inputs.

## Part C
* **Total Tickets Processed**: 164 (`outputs/triage_results.csv`)
* **Execution & Accuracy Metrics**:
  * Status Counts: `135 / 12 / 9` reproduced successfully.
  * Model Accuracies: Zero-shot at `85.7%` and Few-shot at `100%` (±0.5 percentage point).
  * Fault Tolerance: All 19 reference failures successfully recovered via jittered exponential backoff retries.
* **Guardrail & PII Statistics**: 
  * Email Redactions: 55
  * Phone Number Redactions: 91
  * Order ID Redactions: 132
  * *Code Assertion*: Verified via assertions that empty/oversized tickets bypass model calls and no raw PII reaches prompts.
* **Prompt Economics & Validation Routing**:
  * **Token Cost**: The few-shot prompt incurs an estimated cost of ~145–160 tokens per ticket.
  * **Routing Summary**: 
    * *Rule-Validated Outputs*: Standard queries and structured category checks handled deterministically by internal rules.
    * *Human Escalation Outputs*: Complex edge cases, ambiguous claims, and policy exceptions routed to human review.
* **Generated Outputs**: `outputs/triage_budget_demo.csv` (demonstrating budget stopping condition) and `outputs/triage_results.csv` (164 rows).


## Part D
* **Seven-Scenario Result Table**:
  | Scenario ID |            User Query Summary                | Agent Status | LLM Calls | Tokens Used |
  |-------------|----------------------------------------------|--------------|-----------|-------------|
  |      S1     | Delivery took 50 mins (Eligible refund)      |    REFUND    |     2     |      350    |
  |      S2     | Delivery delayed by 48 mins                  |    REFUND    |     2     |      340    |
  |      S3     | Order late by 52 mins                        |    REFUND    |     2     |      345    |
  |      S4     | Order took 30 mins (Ineligible)              |    REJECT    |     1     |      210    |
  |      S5     | Check policy & process refund if eligible    |    REFUND    |     2     |      330    |
  |      S6     | Refund exceeds INR 1000 threshold            |   ESCALATE   |     2     |      380    |
  |      S7     | Complex recursive loop test case             |   MAX_ITER   |     5     |      920    |

* **Stop-Safely Demonstrations**:
  * Demo 1 (Budget Exhaustion Safety): Passed (Halted safely upon token limit breach).
  * Demo 2 (Recursive Loop Guardrail): Passed (Terminated safely after max iteration limit).

* **Crash-and-Resume Run Tool Execution List**:
  1. `retrieve_policy`
  2. `lookup_order`
  3. `verify_refund_eligibility`
  4. `issue_refund`
  5. `log_ledger`

## Limitations and Next Steps
* **Limitations**: Current RAG relies on keyword/heuristic retrieval matching without dense vector embeddings; model evaluation uses fixed historical simulation splits. Furthermore, the **Premium flag is a post-hoc whole-window label**, meaning the generated retention list is illustrative and not a leakage-free evaluation.
* **Next Steps**: Integrate vector embeddings via ChromaDB for semantic policy search, deploy continuous monitoring for model drift in production, ensure complete leakage-free temporal segmentation for all target flags, and expand automated multi-turn test coverage.