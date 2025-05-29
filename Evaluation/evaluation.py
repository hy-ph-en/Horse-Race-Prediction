import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, classification_report, brier_score_loss, precision_recall_curve


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
    
    # Handle case where calibrator might be None
    if model_components['calibrator'] is not None:
        y_pred_proba_raw = model_components['calibrator'].predict_proba(model_components['oof_predictions'])[:, 1]
    else:
        # Use meta_model directly on oof_predictions or just use raw oof_predictions
        if 'meta_model' in model_components and model_components['meta_model'] is not None:
            y_pred_proba_raw = model_components['meta_model'].predict_proba(model_components['oof_predictions'])[:, 1]
        else:
            # Fallback: use the mean of oof_predictions as a simple ensemble
            y_pred_proba_raw = model_components['oof_predictions'].mean(axis=1)
    
    # Apply the same softmax transformation used in production
    # Create temporary dataframe for groupby transformation
    temp_df = train_data[['Race_ID']].copy()
    temp_df['pred_raw'] = y_pred_proba_raw
    
    # Apply softmax normalization by race (same as production)
    y_pred_proba = temp_df.groupby('Race_ID')['pred_raw'].transform(
        lambda x: np.exp(x * 6) / np.exp(x * 6).sum()  # Same scale factor as production
    ).values
    
    y_pred_binary = (y_pred_proba > 0.6).astype(int)
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'brier_score': brier_score_loss(y_true, y_pred_proba)
    }
    
    # Diagnostic information
    print("=== MODEL DIAGNOSTICS ===")
    print(f"Predicted probability range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
    print(f"Mean predicted probability: {y_pred_proba.mean():.4f}")
    print(f"Actual win rate: {y_true.mean():.4f}")
    print(f"Predictions > 0.6: {(y_pred_proba > 0.6).sum()} out of {len(y_pred_proba)}")
    print(f"Predictions > 0.5: {(y_pred_proba > 0.5).sum()} out of {len(y_pred_proba)}")
    print(f"Predictions > 0.1: {(y_pred_proba > 0.1).sum()} out of {len(y_pred_proba)}")
    
    # Find optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold (max F1): {optimal_threshold:.4f}")
    
    # Evaluate with optimal threshold
    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
    optimal_accuracy = accuracy_score(y_true, y_pred_optimal)
    print(f"Accuracy with optimal threshold: {optimal_accuracy:.4f}")
    
    # Print evaluation results
    print("\n=== MODEL EVALUATION METRICS ===")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Accuracy (0.6 threshold): {metrics['accuracy']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print("\nClassification Report (0.6 threshold):")
    print(classification_report(y_true, y_pred_binary, zero_division=0))
    print("\nClassification Report (optimal threshold):")
    print(classification_report(y_true, y_pred_optimal, zero_division=0))
    
    # Add optimal threshold metrics to return
    metrics['optimal_threshold'] = optimal_threshold
    metrics['optimal_accuracy'] = optimal_accuracy
    
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
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'brier_score': brier_score_loss(y_true, y_pred_proba)
    }
    
    # Diagnostic information
    print("=== MODEL DIAGNOSTICS ===")
    print(f"Predicted probability range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
    print(f"Mean predicted probability: {y_pred_proba.mean():.4f}")
    print(f"Actual win rate: {y_true.mean():.4f}")
    print(f"Predictions > 0.6: {(y_pred_proba > 0.6).sum()} out of {len(y_pred_proba)}")
    print(f"Predictions > 0.5: {(y_pred_proba > 0.5).sum()} out of {len(y_pred_proba)}")
    print(f"Predictions > 0.1: {(y_pred_proba > 0.1).sum()} out of {len(y_pred_proba)}")
    
    # Find optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold (max F1): {optimal_threshold:.4f}")
    
    print("\n=== MODEL EVALUATION METRICS ===")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, zero_division=0))
    
    metrics['optimal_threshold'] = optimal_threshold
    
    return metrics
