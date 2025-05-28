from xgboost import XGBClassifier

def get_model(**kwargs):
    """
    Returns an XGBoost classifier.
    Pass hyperparameters via kwargs.
    """
    # Improved hyperparameters for horse racing data
    params = {
        'n_estimators': 100,  # Reduced from 500
        'learning_rate': 0.1,  # Increased from 0.05 for faster convergence
        'max_depth': 5,  # Reduced from 6 to prevent overfitting
        'min_child_weight': 3,  # Increased for regularization
        'subsample': 0.8,  # Added row subsampling
        'colsample_bytree': 0.8,  # Added column subsampling
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,  # Reduce warnings
    }
    params.update(kwargs)
    return XGBClassifier(**params)