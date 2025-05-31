import numpy as np
from configuration import Config
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
    # Load configuration for scale factor
    config = Config()
    scale_factor = config.softmax_scale_factor
        
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
    
    # Apply the same softmax transformation used in production - now configurable
    print(f"Using scale factor {scale_factor} for evaluation (matching production settings)")
    # Create temporary dataframe for groupby transformation
    temp_df = train_data[['Race_ID']].copy()
    temp_df['pred_raw'] = y_pred_proba_raw
    
    # Apply softmax normalization by race (same as production) - now configurable
    y_pred_proba = temp_df.groupby('Race_ID')['pred_raw'].transform(
        lambda x: np.exp(x * scale_factor) / np.exp(x * scale_factor).sum()
    ).values
    
    y_pred_binary = (y_pred_proba > 0.6).astype(int)
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'brier_score': brier_score_loss(y_true, y_pred_proba)
    }
    
    # Further Model Information
    print("=== MODEL INFORMATION ===")
    print(f"Predicted probability range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
    print(f"Mean predicted probability: {y_pred_proba.mean():.4f}")
    print(f"Actual win rate: {y_true.mean():.4f}")
    print(f"Predictions > 0.6: {(y_pred_proba > 0.6).sum()} out of {len(y_pred_proba)}")
    print(f"Predictions > 0.5: {(y_pred_proba > 0.5).sum()} out of {len(y_pred_proba)}")
    print(f"Predictions > 0.1: {(y_pred_proba > 0.1).sum()} out of {len(y_pred_proba)}")
    
    # Run detailed diagnostics if configured
    if config.show_detailed_diagnostics:
        # Import and run detailed diagnostics
        try:
            from .diagnostics import run_detailed_diagnostics
            
            # Ensure we have the required model components
            if ('meta_model' in model_components and model_components['meta_model'] is not None and
                'oof_predictions' in model_components and model_components['oof_predictions'] is not None):
                
                # Create proper dataframes for diagnostic function
                temp_train_df = train_data[['Race_ID', 'win']].copy()
                temp_test_df = train_data[['Race_ID']].copy()
                temp_test_df['pred_raw'] = y_pred_proba_raw
                
                run_detailed_diagnostics(
                    train_df=temp_train_df,
                    test_df=temp_test_df, 
                    meta_model=model_components['meta_model'],
                    oof_preds=model_components['oof_predictions'],
                    group_col='Race_ID',
                    y_pred_proba=y_pred_proba,
                    y_true=y_true
                )
            else:
                print("Missing required model components for detailed diagnostics")
        except Exception as e:
            print(f"Could not run detailed diagnostics: {e}")
    else:
        # Show basic high-confidence analysis
        print("\n=== HIGH CONFIDENCE PREDICTION ANALYSIS ===")
        
        # Analyze predictions >0.5
        high_conf_mask = y_pred_proba > 0.5
        if high_conf_mask.sum() > 0:
            high_conf_actual_wins = y_true[high_conf_mask].sum()
            print(f"Horses predicted >0.5: {high_conf_mask.sum()}")
            print(f"Of these, actual winners: {high_conf_actual_wins}")
            print(f"Hit rate for >0.5 predictions: {high_conf_actual_wins/high_conf_mask.sum():.3f}")
            
            # Show confidence distribution for these predictions
            high_conf_probs = y_pred_proba[high_conf_mask]
            print(f"Confidence range for >0.5 predictions: {high_conf_probs.min():.3f} to {high_conf_probs.max():.3f}")

        # Analyze predictions >0.6
        very_high_conf_mask = y_pred_proba > 0.6
        if very_high_conf_mask.sum() > 0:
            very_high_conf_actual_wins = y_true[very_high_conf_mask].sum()
            print(f"Horses predicted >0.6: {very_high_conf_mask.sum()}")
            print(f"Of these, actual winners: {very_high_conf_actual_wins}")
            print(f"Hit rate for >0.6 predictions: {very_high_conf_actual_wins/very_high_conf_mask.sum():.3f}")

        # Basic winner analysis
        actual_winners_mask = y_true == 1
        if actual_winners_mask.sum() > 0:
            winner_probs = y_pred_proba[actual_winners_mask]
            print(f"\nActual winners ({actual_winners_mask.sum()} total):")
            print(f"  Confidence range: {winner_probs.min():.3f} to {winner_probs.max():.3f}")
            print(f"  Mean confidence: {winner_probs.mean():.3f}")
            print(f"  Winners with >0.5 confidence: {(winner_probs > 0.5).sum()}")
            print(f"  Winners with >0.3 confidence: {(winner_probs > 0.3).sum()}")
            print(f"  Winners with >0.1 confidence: {(winner_probs > 0.1).sum()}")

    # Market comparison (if available) - always show this
    if 'betfairSP' in train_data.columns:
        print(f"\n=== MARKET COMPARISON ===")
        # Add market favorites analysis
        market_favorites = train_data.groupby('Race_ID')['betfairSP'].transform('min') == train_data['betfairSP']
        market_fav_wins = y_true[market_favorites].sum()
        print(f"Market favorites: {market_favorites.sum()}")
        print(f"Market favorite wins: {market_fav_wins}")
        print(f"Market favorite hit rate: {market_fav_wins/market_favorites.sum():.3f}")
        
        # Compare our high confidence picks vs market favorites
        our_picks = y_pred_proba > 0.5
        both_picks = our_picks & market_favorites
        print(f"Overlap (our >0.5 AND market favorite): {both_picks.sum()}")
        if both_picks.sum() > 0:
            print(f"Hit rate when we agree with market: {y_true[both_picks].sum()/both_picks.sum():.3f}")
            
        # What about when we disagree?
        our_picks_not_fav = our_picks & ~market_favorites
        if our_picks_not_fav.sum() > 0:
            print(f"Our >0.5 picks that AREN'T market favorites: {our_picks_not_fav.sum()}")
            print(f"Hit rate for non-favorite picks: {y_true[our_picks_not_fav].sum()/our_picks_not_fav.sum():.3f}")

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
