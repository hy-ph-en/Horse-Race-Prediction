from sklearn.linear_model import LogisticRegression

def get_model(**kwargs):
    """
    Returns a Logistic Regression classifier.
    Pass hyperparameters via kwargs.
    """
    params = {
        'penalty': 'elasticnet',
        'C': 0.1,
        'l1_ratio': 0.5,
        'solver': 'saga',
        'class_weight': {0: 1, 1: 4},  # Manual class weight - less aggressive than 'balanced'
        'max_iter': 2000,
        'random_state': 42,
    }
    params.update(kwargs)
    return LogisticRegression(**params)