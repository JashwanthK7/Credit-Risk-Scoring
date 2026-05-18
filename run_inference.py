import os
import joblib
import pandas as pd
import numpy as np

def predict_online_loan_application():
    model_path = "artifacts/best_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production artifact missing at {model_path}. Run main.py first.")
        
    print("Loading production serialized champion pipeline...")
    production_pipeline = joblib.load(model_path)
    
    # Simulate an incoming application stream (contains missing fields to test stability)
    incoming_data = pd.DataFrame([{
        "person_age": 28,
        "person_income": 55000,
        "person_emp_length": np.nan,    # Missing entry test
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
    
    # Extract prediction probability matrices
    risk_probability = production_pipeline.predict_proba(incoming_data)[0, 1]
    binary_decision = production_pipeline.predict(incoming_data)[0]
    
    print("\n=============================================")
    print("        PRODUCTION INFERENCE REPORT          ")
    print("=============================================")
    print(f"Calculated Default Probability : {risk_probability * 100:.2f}%")
    
    # Establish a risk threshold matching compliance constraints
    compliance_threshold = 0.40
    if risk_probability >= compliance_threshold:
        print("Final Underwriting Status      : REJECTED (High Credit Risk Profile)")
    else:
        print("Final Underwriting Status      : APPROVED (Acceptable Portfolio Risk)")
    print("=============================================")

if __name__ == "__main__":
    predict_online_loan_application()