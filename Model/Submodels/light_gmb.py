from lightgbm import LGBMClassifier

def get_model(**kwargs):
    """
    Returns a LightGBM classifier.
    Pass hyperparameters via kwargs.
    """
    # Improved hyperparameters for horse racing data
    # Reduced complexity to avoid overfitting and training warnings
    params = {
        'n_estimators': 100,  # Reduced from 500
        'learning_rate': 0.1,  # Increased from 0.05 for faster convergence
        'num_leaves': 15,  # Reduced from 31 to prevent overfitting
        'max_depth': 6,  # Added depth limit
        'min_child_samples': 20,  # Increased minimum samples per leaf
        'min_child_weight': 0.001,  # Added regularization
        'subsample': 0.8,  # Added row subsampling
        'colsample_bytree': 0.8,  # Added column subsampling
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbosity': -1,  # Reduce warnings
        'force_row_wise': True,  # Avoid threading issues
    }
    params.update(kwargs)
    return LGBMClassifier(**params)