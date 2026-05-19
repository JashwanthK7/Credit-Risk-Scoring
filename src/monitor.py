import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from src.data_ingestion import run_ingestion, load_config

def generate_drift_report():
    print("\n=============================================")
    print("       GENERATING DATA DRIFT REPORT          ")
    print("=============================================")
    
    config = load_config()
    
    # 1. Load our data splits
    # We will use Train as our Reference and Test as our Current Production data
    X_train, _, X_test, _, _, _ = run_ingestion()

    reference_data = X_train.copy()
    current_data = X_test.copy()

    # 2. Simulate Macroeconomic Drift
    # Let us artificially inflate the income in our Current data by 40% 
    # to simulate a massive economic shift over a few years.
    print("Simulating macroeconomic income inflation in current data...")
    current_data['person_income'] = current_data['person_income'] * 1.40

    # 3. Initialize and run the Evidently AI Drift Report
    print("Calculating statistical drift across all features...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)

    # 4. Save the interactive dashboard using the config path
    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, 'data_drift_dashboard.html')
    drift_report.save_html(report_path)
    
    print(f"Drift Analysis complete! Interactive dashboard saved to: {report_path}")

if __name__ == '__main__':
    generate_drift_report()