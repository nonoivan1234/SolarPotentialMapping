import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


def _recall_class2(y_true, y_pred):
    per_class = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return float(per_class[2])


# class_weight={0:1, 1:1, 2:2}: 高潛力地區(Class 2)的誤分類懲罰加倍，對應廣告投放 KPI
MODELS = {
    'LogisticRegression': LogisticRegression(
        max_iter=1000, class_weight={0: 1, 1: 1, 2: 2}
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight={0: 1, 1: 1, 2: 2}
    ),
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

METRICS = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'roc_auc_ovr': 'roc_auc_ovr',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'recall_class2': make_scorer(_recall_class2),
}

PARAM_GRIDS = {
    'RandomForest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', 0.3],
        'min_samples_leaf': [1, 2, 5],
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
    },
}

TWOSTAGE_MODELS = {
    'RandomForest': {
        'classifier': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            oob_score=True,
        ),
        'regressor': RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            oob_score=True,
        ),
    },
    'XGBoost': {
        'classifier': XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
        ),
        'regressor': XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
        ),
    },
    'GradientBoosting': {
        'classifier': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
        'regressor': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
    },
    'Linear': {
        'classifier': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ]),
        'regressor': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Ridge()),
        ]),
    },
}


TWOSTAGE_PARAM_GRIDS = {
    'RandomForest': {
        'classifier': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 5],
        },
        'regressor': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2', 0.3],
            'min_samples_leaf': [1, 2, 5],
        },
    },
    'XGBoost': {
        'classifier': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        },
        'regressor': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        },
    },
    'GradientBoosting': {
        'classifier': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
        },
        'regressor': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
        },
    },
}


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)


def train_and_evaluate(X, y, model, cv=5, metrics=None, scale_lr=False):
    if metrics is None:
        metrics = METRICS
    if scale_lr and isinstance(model, LogisticRegression):
        # Pipeline ensures scaler is re-fitted on each training fold, preventing data leakage
        eval_model = Pipeline([('scaler', StandardScaler()), ('clf', clone(model))])
    else:
        eval_model = clone(model)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_validate(eval_model, X, y, cv=skf, scoring=metrics, return_train_score=False)
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


def evaluate_binary_classifier(model, X_train, X_test, y_train, y_test):
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
    }
    if hasattr(fitted, 'predict_proba'):
        y_prob = fitted.predict_proba(X_test)[:, 1]
        result['roc_auc'] = roc_auc_score(y_test, y_prob)
    return fitted, result


def evaluate_regressor(model, X_train, X_test, y_train, y_test):
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    result = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': mean_squared_error(y_test, y_pred) ** 0.5,
        'r2': r2_score(y_test, y_pred),
    }
    return fitted, result


def cross_validate_twostage(clf_model, reg_model, X_raw, y_installed, y_density, cv=10):
    """10-fold full-pipeline CV matching DeepSolar paper protocol.

    Imputation, Stage 1 classifier, and Stage 2 regressor are all re-fitted inside
    each fold so no information leaks from the test fold.
    StratifiedKFold preserves the installed/not-installed ratio across folds (Ch12).
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    r2_scores, stage1_f1_scores = [], []
    for train_idx, test_idx in kf.split(X_raw, y_installed):
        X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
        imp = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)

        y_inst_tr = y_installed[train_idx]
        y_inst_te = y_installed[test_idx]
        y_dens_tr = y_density[train_idx]
        y_dens_te = y_density[test_idx]

        clf = clone(clf_model)
        clf.fit(X_tr, y_inst_tr)

        pos_mask = y_inst_tr == 1
        reg = clone(reg_model)
        reg.fit(X_tr[pos_mask], np.log1p(y_dens_tr[pos_mask]))

        inst_pred = clf.predict(X_te)
        y_pred_log = np.zeros(len(test_idx))
        if inst_pred.any():
            y_pred_log[inst_pred == 1] = reg.predict(X_te[inst_pred == 1])
        y_pred_density = np.expm1(y_pred_log)

        r2_scores.append(r2_score(y_dens_te, y_pred_density))
        stage1_f1_scores.append(f1_score(y_inst_te, inst_pred, zero_division=0))

    return {
        'cv_r2_mean': float(np.mean(r2_scores)),
        'cv_r2_std': float(np.std(r2_scores)),
        'cv_stage1_f1_mean': float(np.mean(stage1_f1_scores)),
        'cv_stage1_f1_std': float(np.std(stage1_f1_scores)),
    }


def tune_twostage(model_name, X, y_installed, y_density, n_iter=20, cv=5):
    """
    Sequential hyperparameter tuning for the two-stage model (Appendix D).
    Stage 1: RandomizedSearchCV, scoring=f1 (binary classification).
    Stage 2: RandomizedSearchCV, scoring=r2, trained only on installed tracts.
    Linear is excluded — Pipeline param names require prefix (clf__C, reg__alpha).
    """
    if model_name not in TWOSTAGE_PARAM_GRIDS:
        raise ValueError(
            f'{model_name} has no two-stage parameter grid. '
            f'Available: {list(TWOSTAGE_PARAM_GRIDS.keys())}'
        )
    grids = TWOSTAGE_PARAM_GRIDS[model_name]
    base_models = TWOSTAGE_MODELS[model_name]

    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    clf_search = RandomizedSearchCV(
        clone(base_models['classifier']),
        param_distributions=grids['classifier'],
        n_iter=n_iter,
        cv=skf,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        return_train_score=False,
        verbose=1,
    )
    clf_search.fit(X_imp, y_installed)
    print(f'Stage 1 best F1 : {clf_search.best_score_:.4f}')
    print(f'Stage 1 best params: {clf_search.best_params_}')

    pos_mask = y_installed == 1
    kf_reg = KFold(n_splits=cv, shuffle=True, random_state=42)
    reg_search = RandomizedSearchCV(
        clone(base_models['regressor']),
        param_distributions=grids['regressor'],
        n_iter=n_iter,
        cv=kf_reg,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        return_train_score=False,
        verbose=1,
    )
    reg_search.fit(X_imp[pos_mask], np.log1p(y_density[pos_mask]))
    print(f'Stage 2 best R² (log1p): {reg_search.best_score_:.4f}')
    print(f'Stage 2 best params: {reg_search.best_params_}')

    return clf_search.best_estimator_, reg_search.best_estimator_, {
        'stage1_best_f1': float(clf_search.best_score_),
        'stage1_best_params': clf_search.best_params_,
        'stage2_best_r2_log1p': float(reg_search.best_score_),
        'stage2_best_params': reg_search.best_params_,
    }


def tune_model(X, y, model_name, n_iter=30, cv=5, scoring='recall_class2'):
    """
    RandomizedSearchCV over PARAM_GRIDS[model_name].
    scoring can be any key in METRICS or a sklearn scorer string.
    """
    if model_name not in PARAM_GRIDS:
        raise ValueError(
            f'{model_name} has no parameter grid. '
            f'Available: {list(PARAM_GRIDS.keys())}'
        )
    _scoring = METRICS.get(scoring, scoring) if isinstance(scoring, str) else scoring
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        clone(MODELS[model_name]),
        param_distributions=PARAM_GRIDS[model_name],
        n_iter=n_iter,
        cv=skf,
        scoring=_scoring,
        random_state=42,
        n_jobs=-1,
        return_train_score=False,
        verbose=1,
    )
    search.fit(X, y)
    print(f'Best {scoring}: {search.best_score_:.4f}')
    print(f'Best params: {search.best_params_}')
    return search.best_estimator_, search.best_params_, search.best_score_
