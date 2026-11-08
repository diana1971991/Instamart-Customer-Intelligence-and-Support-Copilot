import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "instamart.db")
SQL_PATH = os.path.join("sql", "queries.sql")

def run_queries():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Please make sure clean_data.py has been run first.")
        return

    if not os.path.exists(SQL_PATH):
        print(f"SQL file not found at {SQL_PATH}.")
        return

    conn = sqlite3.connect(DB_PATH)

    with open(SQL_PATH, 'r') as f:
        sql_content = f.read()

    queries = [q.strip() for q in sql_content.split(";") if q.strip()]

    for i, query in enumerate(queries, 1):
        print(f"\n------- Executing Query {i} -------")
        try:
            df = pd.read_sql_query(query, conn)
            print(f"Result Shape: {df.shape}")
            print(df)
        except Exception as e:
            print(f"Error running Query {i}: {e}")

    conn.close()

if __name__ == "__main__":
    run_queries()