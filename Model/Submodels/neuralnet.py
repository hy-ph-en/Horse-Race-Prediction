from sklearn.neural_network import MLPClassifier

def get_model(**kwargs):
    """
    Returns a Neural Network classifier (MLP).
    Pass hyperparameters via kwargs.
    """
    params = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 1e-4,
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'random_state': 42,
    }
    params.update(kwargs)
    return MLPClassifier(**params)