from sklearn.neural_network import MLPClassifier

def get_model(**kwargs):
    """
    Returns a Neural Network classifier (MLP).
    Pass hyperparameters via kwargs.
    """
    params = {
        'hidden_layer_sizes': (200, 100, 50),  # Deeper network for better pattern recognition
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 1e-5,  # Reduced regularization
        'learning_rate_init': 0.001,
        'max_iter': 1000,  # Increased iterations
        'early_stopping': True,  # Prevent overfitting
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
        'random_state': 42,
    }
    params.update(kwargs)
    return MLPClassifier(**params)