import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, fbeta_score, classification_report
from sklearn.pipeline import Pipeline
from src.pipeline import build_preprocessing_pipeline
from src.data_ingestion import run_ingestion, load_config

def evaluate_model(model, X_train, y_train, X_test, y_test, preprocessor, name):
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    clf_pipeline.fit(X_train, y_train)
    
    y_probs = clf_pipeline.predict_proba(X_test)[:, 1]
    y_preds = clf_pipeline.predict(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    f2 = fbeta_score(y_test, y_preds, beta=2)
    
    print(f'\n=== {name} Baseline Report ===')
    print(classification_report(y_test, y_preds))
    
    return {'Model': name, 'PR-AUC': round(pr_auc, 4), 'F2-Score': round(f2, 4)}

def main():
    config = load_config()
    X_train, X_test, y_train, y_test = run_ingestion()
    preprocessor = build_preprocessing_pipeline(config)
    
    baselines = {
        'Logistic Regression (L2)': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=8, random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    }
    
    results = []
    for name, model in baselines.items():
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, preprocessor, name)
        results.append(metrics)
        
    print('\n=============================================')
    print('         BASELINE MODEL COMPARISON           ')
    print('=============================================')
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == '__main__':
    main()