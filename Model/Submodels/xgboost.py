from xgboost import XGBClassifier

def get_model(**kwargs):
    """
    Returns an XGBoost classifier.
    Pass hyperparameters via kwargs.
    """
    # Improved hyperparameters for horse racing data with class imbalance handling
    params = {
        'n_estimators': 200,  # Increased for better learning
        'learning_rate': 0.05,  # Reduced for more careful learning
        'max_depth': 7,  # Increased for better pattern recognition
        'min_child_weight': 1,  # Reduced for more granular splits
        'subsample': 0.8,  # Added row subsampling
        'colsample_bytree': 0.8,  # Added column subsampling
        'reg_alpha': 0.01,  # Reduced regularization
        'reg_lambda': 0.01,  # Reduced regularization
        'scale_pos_weight': 4.0,  # Reduced from 8.5 - less aggressive class balancing
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,  # Reduce warnings
    }
    params.update(kwargs)
    return XGBClassifier(**params)