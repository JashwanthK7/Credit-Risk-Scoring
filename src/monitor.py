import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from data_ingestion import run_ingestion

def generate_drift_report():
    print("\n=============================================")
    print("       GENERATING DATA DRIFT REPORT          ")
    print("=============================================")
    
    # 1. Load our data splits
    # We will use Train as our "Reference" and Test as our "Current Production" data
    X_train, _, X_test, _, _, _ = run_ingestion()

    reference_data = X_train.copy()
    current_data = X_test.copy()

    # 2. Simulate Macroeconomic Drift
    # Let's artificially inflate the income in our "Current" data by 40% 
    # to simulate a massive economic shift over a few years.
    print("Simulating macroeconomic income inflation in current data...")
    current_data['person_income'] = current_data['person_income'] * 1.40

    # 3. Initialize and run the Evidently AI Drift Report
    print("Calculating statistical drift across all features...")
    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=reference_data, current_data=current_data)

    # 4. Save the interactive dashboard
    os.makedirs("../artifacts", exist_ok=True)
    report_path = "../artifacts/data_drift_dashboard.html"
    snapshot.save_html(report_path)
    
    print(f"Drift Analysis complete! Interactive dashboard saved to: {report_path}")

if __name__ == "__main__":
    generate_drift_report()