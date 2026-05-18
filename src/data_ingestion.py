import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_raw_dataset():
    raw_path = 'data/raw/credit_risk_dataset.csv'
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing base dataset file at {raw_path}")
    return pd.read_csv(raw_path)

def run_ingestion():
    config = load_config()
    df = load_raw_dataset()
    
    X = df.drop(columns=[config['data']['target']])
    y = df[config['data']['target']]
    
    test_ratio = config['data']['test_size']
    val_ratio = config['data']['val_size']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_ratio, 
        stratify=y, 
        random_state=config['data']['random_state']
    )
    
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=relative_val_ratio, 
        stratify=y_temp, 
        random_state=config['data']['random_state']
    )
    
    print(f"Data ingested. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    run_ingestion()