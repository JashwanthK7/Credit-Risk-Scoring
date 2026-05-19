import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from src.data_ingestion import run_ingestion, load_config

def generate_drift_report():
    print('\n=============================================')
    print('       GENERATING DATA DRIFT REPORT          ')
    print('=============================================')
    
    config = load_config()
    X_train, _, X_test, _, _, _ = run_ingestion()

    reference_data = X_train.copy()
    current_data = X_test.copy()

    print('Simulating macroeconomic income inflation in current data...')
    current_data['person_income'] = current_data['person_income'] * 1.40

    print('Calculating statistical drift across all features...')
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)

    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, 'data_drift_dashboard.html')
    drift_report.save_html(report_path)
    
    print(f'Drift Analysis complete! Interactive dashboard saved to: {report_path}')

if __name__ == '__main__':
    generate_drift_report()