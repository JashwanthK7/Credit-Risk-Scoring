import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [formData, setFormData] = useState({
    person_age: 28,
    person_income: 55000,
    person_emp_length: 5,
    loan_amnt: 12000,
    loan_int_rate: 11.5,
    loan_percent_income: 0.22,
    cb_person_cred_hist_length: 5,
    person_home_ownership: 'RENT',
    loan_intent: 'DEBTCONSOLIDATION',
    loan_grade: 'B',
    cb_person_default_on_file: 'N'
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const parameterGuide = {
    Age: 'Applicant age in years. Must be at least 18.',
    Annual_Income: 'Total yearly income in dollars.',
    Employment_Length: 'Number of years employed at the current job.',
    Loan_Amount: 'The total amount of money requested.',
    Interest_Rate: 'The interest rate assigned to the loan as a percentage.',
    Loan_to_Income_Ratio: 'The loan amount divided by the annual income.',
    Credit_History_Length: 'The number of years since the applicant opened their first credit line.',
    Home_Ownership: 'Current living situation (Rent, Mortgage, Own, or Other).',
    Loan_Intent: 'The primary purpose or reason for requesting the loan.',
    Loan_Grade: 'A risk grade assigned by the institution, where A is the safest and G is the riskiest.',
    Historical_Default: 'Indicates if the applicant has a previous default on record.'
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value === '' ? '' : isNaN(value) ? value : Number(value)
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while fetching the prediction.');
    }
  };

  return (
    <div style={{ display: 'flex', maxWidth: '1000px', margin: '0 auto', padding: '20px', fontFamily: 'sans-serif', gap: '40px' }}>

      <style>
        {`
          label { font-size: 16px; font-weight: bold; margin-top: 5px; }
          input, select { font-size: 16px; padding: 8px; border-radius: 4px; border: 1px solid #ccc; }
          button { font-size: 16px; font-weight: bold; }
          .guide-text { font-size: 16px; }
        `}
      </style>

      <div style={{ flex: '1' }}>
        <h2>Credit Risk Application</h2>
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

          <label>Age:</label>
          <input type='number' name='person_age' value={formData.person_age} onChange={handleChange} required />

          <label>Annual Income:</label>
          <input type='number' name='person_income' value={formData.person_income} onChange={handleChange} required />

          <label>Employment Length (years):</label>
          <input type='number' name='person_emp_length' value={formData.person_emp_length} onChange={handleChange} />

          <label>Loan Amount:</label>
          <input type='number' name='loan_amnt' value={formData.loan_amnt} onChange={handleChange} required />

          <label>Interest Rate:</label>
          <input type='number' step='0.1' name='loan_int_rate' value={formData.loan_int_rate} onChange={handleChange} />

          <label>Loan to Income Ratio:</label>
          <input type='number' step='0.01' name='loan_percent_income' value={formData.loan_percent_income} onChange={handleChange} required />

          <label>Credit History Length (years):</label>
          <input type='number' name='cb_person_cred_hist_length' value={formData.cb_person_cred_hist_length} onChange={handleChange} required />

          <label>Home Ownership:</label>
          <select name='person_home_ownership' value={formData.person_home_ownership} onChange={handleChange}>
            <option value='RENT'>Rent</option>
            <option value='MORTGAGE'>Mortgage</option>
            <option value='OWN'>Own</option>
            <option value='OTHER'>Other</option>
          </select>

          <label>Loan Intent:</label>
          <select name='loan_intent' value={formData.loan_intent} onChange={handleChange}>
            <option value='PERSONAL'>Personal</option>
            <option value='EDUCATION'>Education</option>
            <option value='MEDICAL'>Medical</option>
            <option value='VENTURE'>Venture</option>
            <option value='HOMEIMPROVEMENT'>Home Improvement</option>
            <option value='DEBTCONSOLIDATION'>Debt Consolidation</option>
          </select>

          <label>Loan Grade:</label>
          <select name='loan_grade' value={formData.loan_grade} onChange={handleChange}>
            {['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(grade => (
              <option key={grade} value={grade}>{grade}</option>
            ))}
          </select>

          <label>Historical Default on File:</label>
          <select name='cb_person_default_on_file' value={formData.cb_person_default_on_file} onChange={handleChange}>
            <option value='Y'>Yes</option>
            <option value='N'>No</option>
          </select>

          <button type='submit' style={{ padding: '12px', marginTop: '15px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
            Evaluate Application
          </button>
        </form>

        {error && <div style={{ color: 'red', marginTop: '20px', fontSize: '16px' }}>Error: {error}</div>}

        {result && (
          <div style={{ marginTop: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '4px', backgroundColor: '#e9ecef', fontSize: '16px' }}>
            <h3>Underwriting Decision</h3>
            <p>Status: <strong>{result.risk_status}</strong></p>
            <p>Default Probability: {(result.default_probability * 100).toFixed(2)}%</p>
            <p>Compliance Threshold: {(result.compliance_threshold * 100).toFixed(2)}%</p>
          </div>
        )}
      </div>

      <div style={{ flex: '1', backgroundColor: '#f8f9fa', padding: '20px', borderRadius: '8px', height: 'fit-content', marginTop: '65px' }}>
        <h3>Parameter Guide</h3>
        <ul style={{ listStyleType: 'none', padding: 0 }}>
          {Object.entries(parameterGuide).map(([key, desc]) => (
            <li key={key} className="guide-text" style={{ marginBottom: '15px', lineHeight: '1.4' }}>
              <strong>{key.replace(/_/g, ' ')}:</strong> <br /> {desc}
            </li>
          ))}
        </ul>
      </div>

    </div>
  );
}

export default App;