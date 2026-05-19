import os
import pandas as pd
import numpy as np
from src.data_ingestion import load_raw_dataset, load_config

def run_eda():
    print('\n=============================================')
    print('       RUNNING ADVANCED DATA PROFILING       ')
    print('=============================================')
    
    config = load_config()
    df = load_raw_dataset(config)
    target = config['data']['target']
    num_features = config['features']['numerical']
    cat_features = config['features']['categorical']
    
    os.makedirs(config['paths']['artifacts_dir'], exist_ok=True)
    report_path = os.path.join(config['paths']['artifacts_dir'], 'eda_report.md')
    
    with open(report_path, 'w') as f:
        f.write('# Credit Risk Pipeline: Exploratory Data Analysis Report\n\n')
        
        f.write('## 1. Dataset Dimensions\n')
        f.write(f'* Total Records: {df.shape[0]}\n')
        f.write(f'* Total Features: {df.shape[1] - 1}\n\n')
        
        f.write('## 2. Target Class Distribution (loan_status)\n')
        counts = df[target].value_counts()
        pcts = df[target].value_counts(normalize=True) * 100
        f.write('| Risk Class | Count | Percentage |\n| --- | --- | --- |\n')
        f.write(f'| Non-Default (0) | {counts.get(0, 0)} | {pcts.get(0, 0):.2f}% |\n')
        f.write(f'| Default (1) | {counts.get(1, 0)} | {pcts.get(1, 0):.2f}% |\n\n')
        
        f.write('## 3. Missing Data & Data Completeness\n')
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
        
        f.write('| Feature Name | Missing Values | Percentage Missing |\n| --- | --- | --- |\n')
        for col, row in missing_df.iterrows():
            f.write(f'| {col} | {row["Missing"]} | {row["Percentage"]:.4f}% |\n')
        f.write('\n')
        
        f.write('## 4. Advanced Numerical Feature Distributions\n')
        f.write('| Feature | Mean | Median | Std Dev | Skewness | Kurtosis |\n| --- | --- | --- | --- | --- | --- |\n')
        for col in num_features:
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            skew_val = df[col].skew()
            kurt_val = df[col].kurt()
            f.write(f'| {col} | {mean_val:.2f} | {median_val:.2f} | {std_val:.2f} | {skew_val:.2f} | {kurt_val:.2f} |\n')
        f.write('\n')
        
        f.write('## 5. Linear Correlation Matrix (Numerical vs Target)\n')
        corr_matrix = df[num_features + [target]].corr()[target].sort_values(ascending=False)
        f.write('| Feature Name | Linear Correlation with Target |\n| --- | --- |\n')
        for col, val in corr_matrix.items():
            if col != target:
                f.write(f'| {col} | {val:.4f} |\n')
        f.write('\n')
        
        f.write('## 6. Categorical Feature Risk Profiling\n')
        for col in cat_features:
            f.write(f'### Default Rates by {col}\n')
            risk_profile = df.groupby(col)[target].agg(['count', 'mean']).reset_index()
            risk_profile['mean'] = risk_profile['mean'] * 100
            f.write(f'| {col} | Total Applications | Default Rate (%) |\n| --- | --- | --- |\n')
            for _, row in risk_profile.iterrows():
                f.write(f'| {row[col]} | {row["count"]} | {row["mean"]:.2f}% |\n')
            f.write('\n')
            
    print(f'Data profiling complete. Comprehensive analysis report saved to: {report_path}')

if __name__ == '__main__':
    run_eda()