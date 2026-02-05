from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_logistic_model():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

def get_xgb_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
