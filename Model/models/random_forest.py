from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
import logging

def train_rf(X_train, y_train):
    logging.info("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_rf(model, X, y):
    logging.info("Evaluating Random Forest...")
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    mcc = matthews_corrcoef(y, preds)
    auc = roc_auc_score(y, probs)
    logging.info(f"Random Forest MCC: {mcc:.4f}, AUC: {auc:.4f}")
    return mcc, auc
