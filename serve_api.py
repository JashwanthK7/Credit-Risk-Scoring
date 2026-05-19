import os
import urllib.request
import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
compliance_threshold = config['business_logic']['compliance_threshold']

model_path = 'artifacts/best_model.pkl'
model_url = os.environ.get('MODEL_URL')

if model_url and not os.path.exists(model_path):
    os.makedirs('artifacts', exist_ok=True)
    print('Downloading production model from remote registry...')
    urllib.request.urlretrieve(model_url, model_path)

try:
    production_pipeline = joblib.load(model_path)
except Exception as e:
    production_pipeline = None
    print(f'Failed to load model artifact: {e}')

API_KEY_NAME = 'X-API-Key'
API_KEY = os.environ.get('CREDIT_RISK_API_KEY', 'default-dev-key-123')
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Invalid or missing API Key'
    )

class LoanApplication(BaseModel):
    person_age: int = Field(..., gt=17, lt=100)
    person_income: int = Field(..., gt=0)
    person_emp_length: float | None = Field(None, ge=0, lt=60)
    loan_amnt: int = Field(..., gt=0)
    loan_int_rate: float | None = Field(None, ge=0, lt=30)
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_cred_hist_length: int = Field(..., ge=0)
    person_home_ownership: str = Field(...)
    loan_intent: str = Field(...)
    loan_grade: str = Field(...)
    cb_person_default_on_file: str = Field(...)

app = FastAPI(title='Credit Risk Scoring API', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
def health_check():
    return {'status': 'Online', 'message': 'Credit Risk API is actively running.'}

@app.post('/predict')
def predict_risk(application: LoanApplication, api_key: str = Depends(verify_api_key)):
    if production_pipeline is None:
        raise HTTPException(status_code=500, detail='Model pipeline is not loaded.')
        
    input_data = pd.DataFrame([application.model_dump()])
    
    try:
        risk_probability = production_pipeline.predict_proba(input_data)[0, 1]
        status_result = 'REJECTED' if risk_probability >= compliance_threshold else 'APPROVED'
        
        return {
            'default_probability': round(risk_probability, 4),
            'risk_status': status_result,
            'compliance_threshold': compliance_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Prediction error: {str(e)}')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)