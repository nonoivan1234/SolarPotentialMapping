import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, n_jobs=-1),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
    ),
}

METRICS = ['accuracy', 'f1_macro', 'roc_auc_ovr', 'precision_macro', 'recall_macro']


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)


def train_and_evaluate(X, y, model, cv=5, metrics=None, scale_lr=False):
    if metrics is None:
        metrics = METRICS
    if scale_lr and isinstance(model, LogisticRegression):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=skf, scoring=metrics, return_train_score=False)
    result = {k.replace('test_', ''): np.mean(v) for k, v in scores.items() if k.startswith('test_')}
    return result


def compare_models(X, y, models=None, cv=5, scale_lr=True):
    if models is None:
        models = MODELS
    results = []
    for name, model in models.items():
        scores = train_and_evaluate(X, y, model, cv=cv, scale_lr=scale_lr)
        row = {'model': name, **scores}
        results.append(row)
    return pd.DataFrame(results)
