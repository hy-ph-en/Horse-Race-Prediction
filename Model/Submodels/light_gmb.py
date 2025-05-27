from lightgbm import LGBMClassifier

def get_model(**kwargs):
    """
    Returns a LightGBM classifier.
    Pass hyperparameters via kwargs.
    """
    # Example hyperparameters; tune as needed
    params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'objective': 'binary',
        'random_state': 42,
    }
    params.update(kwargs)
    return LGBMClassifier(**params)