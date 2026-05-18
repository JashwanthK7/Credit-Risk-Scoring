import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the production artifact on startup
model_path = 'artifacts/best_model.pkl'
try:
    production_pipeline = joblib.load(model_path)
except Exception as e:
    production_pipeline = None
    print(f"Failed to load model artifact: {e}")

# Define the strict input schema using Pydantic
class LoanApplication(BaseModel):
    person_age: int = Field(..., gt=17, lt=100, description='Applicant age in years')
    person_income: int = Field(..., gt=0, description='Annual income in dollars')
    person_emp_length: float | None = Field(None, ge=0, lt=60, description='Employment length in years')
    loan_amnt: int = Field(..., gt=0, description='Requested loan amount')
    loan_int_rate: float | None = Field(None, ge=0, lt=30, description='Interest rate percentage')
    loan_percent_income: float = Field(..., ge=0, le=1, description='Loan amount divided by income')
    cb_person_cred_hist_length: int = Field(..., ge=0, description='Credit history length in years')
    person_home_ownership: str = Field(..., description='RENT, MORTGAGE, OWN, or OTHER')
    loan_intent: str = Field(..., description='Purpose of the loan')
    loan_grade: str = Field(..., description='Assigned loan grade A through G')
    cb_person_default_on_file: str = Field(..., description='Y or N')

app = FastAPI(title='Credit Risk Scoring API', version='1.0')

@app.get('/')
def health_check():
    return {"status": "Online", "message": "Credit Risk API is actively running."}

@app.post('/predict')
def predict_risk(application: LoanApplication):
    if production_pipeline is None:
        raise HTTPException(status_code=500, detail='Model pipeline is not loaded.')
        
    # Convert validated Pydantic object to a single row DataFrame
    input_data = pd.DataFrame([application.model_dump()])
    
    try:
        # Extract prediction probabilities
        risk_probability = production_pipeline.predict_proba(input_data)[0, 1]
        
        # Apply compliance threshold logic
        compliance_threshold = 0.40
        status = 'REJECTED' if risk_probability >= compliance_threshold else 'APPROVED'
        
        return {
            'default_probability': round(risk_probability, 4),
            'risk_status': status,
            'compliance_threshold': compliance_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)