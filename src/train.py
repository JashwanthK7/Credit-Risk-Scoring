import os
import joblib
import pandas as pd
import numpy as np
import shap
import mlflow
import mlflow.sklearn
import warnings
import optuna

warnings.filterwarnings(action='ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_ingestion import run_ingestion, load_config
from src.pipeline import build_preprocessing_pipeline

def get_feature_names(preprocessor):
    try:
        num_cols = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.transformers_[1][1].named_steps['encoder']
        cat_cols = preprocessor.transformers_[1][2]
        cat_encoded_cols = list(cat_encoder.get_feature_names_out(cat_cols))
        return num_cols + cat_encoded_cols
    except Exception:
        return None

def evaluate_performance(y_true, y_probs, y_preds):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    f2 = fbeta_score(y_true, y_preds, beta=2)
    return round(pr_auc, 4), round(f2, 4)

def run_baselines(X_train_proc, X_val_proc, y_train, y_val, scale_weight_val):
    print("\n=============================================")
    print("       STAGE 1: BASELINE TOURNAMENT (VAL SET)")
    print("=============================================")
    
    baselines = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=8, random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_weight_val),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_weight_val)
    }
    
    results = []
    for name, model in baselines.items():
        model.fit(X_train_proc, y_train)
        y_probs = model.predict_proba(X_val_proc)[:, 1]
        y_preds = model.predict(X_val_proc)
        pr_auc, f2 = evaluate_performance(y_val, y_probs, y_preds)
        results.append({'Model': name, 'Val PR-AUC': pr_auc, 'Val F2-Score': f2})
        
    results_df = pd.DataFrame(results).sort_values(by='Val PR-AUC', ascending=False)
    print(results_df.to_string(index=False))

def run_optuna_tuning(X_train_proc, X_val_proc, y_train, y_val, scale_weight_val):
    print("\n=============================================")
    print("       STAGE 2: BAYESIAN HYPERPARAM TUNING   ")
    print("=============================================")
    
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=100),
            'max_depth': trial.suggest_int('max_depth', 10, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10)
        }
        model = RandomForestClassifier(random_state=42, class_weight='balanced', **params)
        model.fit(X_train_proc, y_train)
        probs = model.predict_proba(X_val_proc)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, probs)
        return auc(recall, precision)

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=100),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0)
        }
        model = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_weight_val, **params)
        model.fit(X_train_proc, y_train)
        probs = model.predict_proba(X_val_proc)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, probs)
        return auc(recall, precision)

    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=100),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
        }
        model = LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_weight_val, **params)
        model.fit(X_train_proc, y_train)
        probs = model.predict_proba(X_val_proc)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, probs)
        return auc(recall, precision)

    print("Optimizing Random Forest Space...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=5)
    print(f"  Best RF Val PR-AUC: {study_rf.best_value:.4f}")
    
    print("Optimizing XGBoost Space...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=5)
    print(f"  Best XGB Val PR-AUC: {study_xgb.best_value:.4f}")
    
    print("Optimizing LightGBM Space...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=5)
    print(f"  Best LGBM Val PR-AUC: {study_lgb.best_value:.4f}")
    
    return study_rf, study_xgb, study_lgb

def select_champion(study_rf, study_xgb, study_lgb, scale_weight_val):
    print("\n=============================================")
    print("       STAGE 3: CHAMPION SELECTION LOGIC     ")
    print("=============================================")
    
    best_boosting_score = max(study_xgb.best_value, study_lgb.best_value)
    rf_score = study_rf.best_value
    compliance_margin = 0.015
    
    if rf_score >= (best_boosting_score - compliance_margin):
        winner_name = 'RandomForest'
        final_model = RandomForestClassifier(random_state=42, class_weight='balanced', **study_rf.best_params)
        print(f">>>> Champion Selected: {winner_name} (Val PR-AUC: {rf_score:.4f})")
    else:
        if study_xgb.best_value >= study_lgb.best_value:
            winner_name = 'XGBoost'
            final_model = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_weight_val, **study_xgb.best_params)
        else:
            winner_name = 'LightGBM'
            final_model = LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_weight_val, **study_lgb.best_params)
        print(f">>>> Champion Selected: {winner_name} (Val PR-AUC: {max(study_xgb.best_value, study_lgb.best_value):.4f})")
        
    return final_model

def final_test_evaluation(champion_pipeline, X_test, y_test):
    print("\n=============================================")
    print("       STAGE 4: HOLDOUT TEST EVALUATION      ")
    print("=============================================")
    y_probs = champion_pipeline.predict_proba(X_test)[:, 1]
    y_preds = champion_pipeline.predict(X_test)
    test_pr_auc, test_f2 = evaluate_performance(y_test, y_probs, y_preds)
    print(f"Unseen Test Set PR-AUC : {test_pr_auc:.4f}")
    print(f"Unseen Test Set F2     : {test_f2:.4f}")

def compute_model_explainability(champion_pipeline, X_test, preprocessor):
    print("\n=============================================")
    print("       STAGE 5: SHAP EXPLAINABILITY (TEST)   ")
    print("=============================================")
    
    estimator = champion_pipeline.named_steps['classifier']
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_test_proc.shape[1])]
        
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=feature_names)
    
    explainer = shap.Explainer(estimator, X_test_proc_df)
    shap_values = explainer(X_test_proc_df, check_additivity=False)
    
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    shap_summary = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP| (Marginal Contribution)': mean_shap
    }).sort_values(by='Mean |SHAP| (Marginal Contribution)', ascending=False)
    
    print(shap_summary.to_string(index=False))

def main_training_pipeline():
    config = load_config()
    X_train, X_val, X_test, y_train, y_val, y_test = run_ingestion()
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight_val = num_neg / num_pos
    
    preprocessor = build_preprocessing_pipeline(config)
    preprocessor.fit(X_train, y_train)
    
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    run_baselines(X_train_proc, X_val_proc, y_train, y_val, scale_weight_val)
    
    study_rf, study_xgb, study_lgb = run_optuna_tuning(X_train_proc, X_val_proc, y_train, y_val, scale_weight_val)
    final_model = select_champion(study_rf, study_xgb, study_lgb, scale_weight_val)
    
    champion_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', final_model)
    ])
    champion_pipeline.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # NEW: MLflow Experiment Tracking
    # ---------------------------------------------------------
    print("\n=============================================")
    print("       STAGE 4: MLFLOW EXPERIMENT LOGGING    ")
    print("=============================================")
    
    # Set the name of the experiment project
    mlflow.set_experiment("Credit_Risk_Optimization")
    
    # Start a tracking run
    with mlflow.start_run(run_name="Champion_Model_Evaluation"):
        # Calculate test metrics
        y_probs = champion_pipeline.predict_proba(X_test)[:, 1]
        y_preds = champion_pipeline.predict(X_test)
        test_pr_auc, test_f2 = evaluate_performance(y_test, y_probs, y_preds)
        
        print(f"Unseen Test Set PR-AUC : {test_pr_auc:.4f}")
        print(f"Unseen Test Set F2     : {test_f2:.4f}")

        # Log the hyperparameters of the winning model
        mlflow.log_params(final_model.get_params())
        
        # Log the final holdout metrics
        mlflow.log_metric("test_pr_auc", test_pr_auc)
        mlflow.log_metric("test_f2", test_f2)
        
        # Securely log the actual model binary inside MLflow
        mlflow.sklearn.log_model(champion_pipeline, "production_pipeline")
        print("Model, parameters, and metrics successfully logged to MLflow.")
    
    # Keep local save for your Docker container
    os.makedirs(config['paths']['artifacts_dir'], exist_ok=True)
    joblib.dump(champion_pipeline, os.path.join(config['paths']['artifacts_dir'], 'best_model.pkl'))
    
    compute_model_explainability(champion_pipeline, X_test, preprocessor)

if __name__ == '__main__':
    main_training_pipeline()