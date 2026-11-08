import subprocess
import sys
import os

def run_step(script_name, description):
    print("=" * 60)
    print(f"RUNNING: {description} ({script_name})")
    print("=" * 60)
    
    script_path = os.path.join("src", script_name)
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    result = subprocess.run([venv_python, script_path])

    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at step: {description}")
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] Completed: {description}\n")

if __name__ == "__main__":
    print("Starting Instamart End-to-End Pipeline Execution...\n")
    
    # Step 1: Data Auditing, Cleaning & Database Generation (Part A)
    run_step("clean_data.py", "PART A: DATA FOUNDATION - AUDIT, CLEAN, & STORE")
    
    # Step 2: SQL Query Verification (Part A)
    run_step("run_queries.py", "SQL Query Verification")

    # Step 3: SQL & EXPLORATORY CROSS-CHECKS (Part A)
    run_step("exploratory_analysis.py", "SQL & EXPLORATORY CROSS-CHECKS")
    
    # Step 4: Customer Segmentation & Lapse Modeling (Part B)
    run_step("CustSeg_LapsePred.py", "PART B: Customer Segmentation & Lapse Prediction")

    # Step 5: Customer Segmentation & Lapse Modeling (Part B)
    run_step("triage.py", "Part C : Support Ticket Triage with Guardrails")

    # Step 5: Customer Segmentation & Lapse Modeling (Part B)
    run_step("policy_rag.py", "Part D : Policy RAG Assistant and Refund Agent ")
    
    print("=" * 70)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)