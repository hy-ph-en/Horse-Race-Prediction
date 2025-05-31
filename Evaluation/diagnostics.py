import numpy as np
import pandas as pd


def run_detailed_diagnostics(train_df, test_df, meta_model, oof_preds, group_col, y_pred_proba=None, y_true=None):
    """
    Run detailed scale factor and threshold diagnostics.
    Separated from main functions to avoid cluttering output.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe 
        meta_model: Trained meta model
        oof_preds: Out-of-fold predictions
        group_col: Group column name
        y_pred_proba: Evaluation probabilities (optional)
        y_true: True labels for evaluation (optional)
    """
    print(f"\n=== DETAILED DIAGNOSTICS ===")
    
    # Scale Factor Diagnostics
    print(f"\n--- SCALE FACTOR ANALYSIS ---")
    scale_factors = [1, 2, 4, 6, 8, 10]
    
    # Get true labels for Brier score calculation (if available)
    if 'win' in train_df.columns:
        # Use training data to approximate impact
        train_preds = meta_model.predict_proba(oof_preds)[:, 1]
        train_df_temp = train_df.copy()
        train_df_temp['pred_raw'] = train_preds
        y_true_train = train_df['win'].values
        
        print(f"Scale Factor Impact Analysis (using training data):")
        for scale in scale_factors:
            train_probs = train_df_temp.groupby(group_col)['pred_raw'].transform(
                lambda x: np.exp(x * scale) / np.exp(x * scale).sum()
            )
            # Calculate Brier score
            brier = np.mean((train_probs - y_true_train) ** 2)
            log_loss_val = -np.mean(y_true_train * np.log(np.clip(train_probs, 1e-15, 1-1e-15)) + 
                                  (1 - y_true_train) * np.log(np.clip(1 - train_probs, 1e-15, 1-1e-15)))
            
            print(f"Scale {scale}: Range {train_probs.min():.4f}-{train_probs.max():.4f}, "
                  f">0.5: {np.sum(train_probs > 0.5)}, Brier: {brier:.4f}, LogLoss: {log_loss_val:.4f}")
    
    # Test on actual test data
    print(f"\nTest Data Scale Factor Analysis:")
    for scale in scale_factors:
        test_probs = test_df.groupby(group_col)['pred_raw'].transform(
            lambda x: np.exp(x * scale) / np.exp(x * scale).sum()
        )
        print(f"Test Scale {scale}: Range {test_probs.min():.4f} to {test_probs.max():.4f}, "
              f"Mean {test_probs.mean():.4f}, >0.5: {np.sum(test_probs > 0.5)}")
    
    # Example race analysis
    if 'pred_raw' in test_df.columns:
        first_race = test_df[test_df[group_col] == test_df[group_col].iloc[0]]
        print(f"\nExample race analysis (Race {first_race[group_col].iloc[0]}):")
        print(f"Raw predictions: {first_race['pred_raw'].values}")
        print(f"Raw range: {first_race['pred_raw'].min():.4f} to {first_race['pred_raw'].max():.4f}")
        print(f"Raw std: {first_race['pred_raw'].std():.4f}")
        
        for scale in [1, 4, 8]:
            race_softmax = np.exp(first_race['pred_raw'].values * scale)
            race_probs = race_softmax / race_softmax.sum()
            print(f"Scale {scale}: {race_probs} (max: {race_probs.max():.4f})")

    # Threshold Analysis (if evaluation data provided)
    if y_pred_proba is not None and y_true is not None:
        print(f"\n--- THRESHOLD ANALYSIS ---")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mask = y_pred_proba > thresh
            if mask.sum() > 0:
                actual_wins = y_true[mask].sum()
                hit_rate = actual_wins / mask.sum()
                recall = actual_wins / y_true.sum() if y_true.sum() > 0 else 0
                print(f"Threshold {thresh}: {mask.sum()} predictions, {actual_wins} winners, "
                      f"Hit rate: {hit_rate:.3f}, Recall: {recall:.3f}")
            else:
                print(f"Threshold {thresh}: 0 predictions")

        # Winner confidence analysis
        actual_winners_mask = y_true == 1
        if actual_winners_mask.sum() > 0:
            winner_probs = y_pred_proba[actual_winners_mask]
            print(f"\nWinner Confidence Analysis ({actual_winners_mask.sum()} total winners):")
            print(f"  Confidence range: {winner_probs.min():.3f} to {winner_probs.max():.3f}")
            print(f"  Mean confidence: {winner_probs.mean():.3f}")
            print(f"  Median confidence: {np.median(winner_probs):.3f}")
            print(f"  Winners with >0.5 confidence: {(winner_probs > 0.5).sum()}")
            print(f"  Winners with >0.3 confidence: {(winner_probs > 0.3).sum()}")
            print(f"  Winners with >0.1 confidence: {(winner_probs > 0.1).sum()}")
            
            # Show distribution of winner confidences
            print(f"  Winner confidence quartiles:")
            print(f"    25th percentile: {np.percentile(winner_probs, 25):.3f}")
            print(f"    50th percentile: {np.percentile(winner_probs, 50):.3f}")
            print(f"    75th percentile: {np.percentile(winner_probs, 75):.3f}")
            print(f"    95th percentile: {np.percentile(winner_probs, 95):.3f}") 