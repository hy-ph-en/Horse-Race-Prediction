from sklearn.linear_model import LogisticRegression

def get_model(**kwargs):
    """
    Returns a Logistic Regression classifier.
    Pass hyperparameters via kwargs.
    """
    params = {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'liblinear',
        'random_state': 42,
    }
    params.update(kwargs)
    return LogisticRegression(**params)