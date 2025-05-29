from lightgbm import LGBMClassifier

def get_model(**kwargs):
    """
    Returns a LightGBM classifier.
    Pass hyperparameters via kwargs.
    """
    # Improved hyperparameters for horse racing data with class imbalance handling
    params = {
        'n_estimators': 200,  # Increased for better learning
        'learning_rate': 0.05,  # Reduced for more careful learning
        'num_leaves': 31,  # Increased for more complexity
        'max_depth': 8,  # Increased depth for better pattern recognition
        'min_child_samples': 10,  # Reduced to allow more granular splits
        'min_child_weight': 0.001,  
        'subsample': 0.8,  
        'colsample_bytree': 0.8,  
        'reg_alpha': 0.01,  # Reduced regularization
        'reg_lambda': 0.01,  # Reduced regularization
        'scale_pos_weight': 8.5,  # Handle class imbalance (approximate ratio of negative to positive)
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbosity': -1,  
        'force_row_wise': True,  
    }
    params.update(kwargs)
    return LGBMClassifier(**params)