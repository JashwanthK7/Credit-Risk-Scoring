import os
import joblib
import pandas as pd
import numpy as np
import yaml

def predict_online_loan_application():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    compliance_threshold = config['business_logic']['compliance_threshold']

    model_path = "artifacts/best_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production artifact missing at {model_path}. Run main.py first.")
        
    print("Loading production serialized champion pipeline...")
    production_pipeline = joblib.load(model_path)
    
    incoming_data = pd.DataFrame([{
        "person_age": 28,
        "person_income": 55000,
        "person_emp_length": np.nan,
        "loan_amnt": 12000,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.22,
        "cb_person_cred_hist_length": 5,
        "person_home_ownership": "RENT",
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "B",
        "cb_person_default_on_file": "N"
    }])
    
    print("\nProcessing streaming transaction payload...")
    risk_probability = production_pipeline.predict_proba(incoming_data)[0, 1]
    
    print("\n=============================================")
    print("        PRODUCTION INFERENCE REPORT          ")
    print("=============================================")
    print(f"Calculated Default Probability : {risk_probability * 100:.2f}%")
    
    if risk_probability >= compliance_threshold:
        print("Final Underwriting Status      : REJECTED (High Credit Risk Profile)")
    else:
        print("Final Underwriting Status      : APPROVED (Acceptable Portfolio Risk)")
    print("=============================================")

if __name__ == "__main__":
    predict_online_loan_application()