import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, classification_report


def evaluate_model(model_components, train_data):
    """
    Evaluate model performance with various metrics using out-of-fold predictions.
    
    Args:
        model_components: Dictionary containing trained model components
        train_data: Training dataframe with target variable
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get out-of-fold predictions and true labels
    y_true = train_data['win'].values
    y_pred_proba = model_components['calibrator'].predict_proba(model_components['oof_predictions'])[:, 1]
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred_binary)
    }
    
    # Print evaluation results
    print("Model Evaluation Metrics:")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary))
    
    return metrics


def evaluate_model_simple(y_true, y_pred_proba, y_pred_binary=None):
    """
    Simple evaluation function for direct predictions.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred_binary: Predicted binary labels (optional)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred_binary)
    }
    
    print("Model Evaluation Metrics:")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary))
    
    return metrics
