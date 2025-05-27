from xgboost import XGBClassifier

def get_model(**kwargs):
    """
    Returns an XGBoost classifier.
    Pass hyperparameters via kwargs.
    """
    params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
    }
    params.update(kwargs)
    return XGBClassifier(**params)