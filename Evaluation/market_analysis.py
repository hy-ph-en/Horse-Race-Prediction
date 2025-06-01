import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from configuration import Config
from Data_Processing.preprocessing import Preprocessing
from Model.stack import model_run
from Evaluation.evaluation import evaluate_model


def analyze_market_comparison(train_data, original_data, y_pred_proba, y_true):
    """
    Comprehensive analysis comparing model predictions against market odds (betfairSP).
    
    Args:
        train_data: Processed DataFrame with race data
        original_data: Original DataFrame with betfairSP column
        y_pred_proba: Model predicted probabilities
        y_true: Actual results (1 for winner, 0 for non-winner)
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MARKET COMPARISON ANALYSIS")
    print("="*80)
    
    # Merge original data to get betfairSP
    # Create a mapping from processed data to original data
    merge_cols = ['Race_ID', 'Horse']  # Unique identifiers
    
    # Get betfairSP from original data
    market_data = original_data[merge_cols + ['betfairSP']].copy()
    
    # Merge with processed data to align with predictions
    combined_data = train_data[merge_cols + ['win']].merge(market_data, on=merge_cols, how='left')
    
    # Check for missing betfairSP values
    missing_odds = combined_data['betfairSP'].isna().sum()
    if missing_odds > 0:
        print(f"Warning: {missing_odds} rows missing betfairSP data. Dropping these rows.")
        valid_mask = combined_data['betfairSP'].notna()
        combined_data = combined_data[valid_mask]
        y_pred_proba = y_pred_proba[valid_mask]
        y_true = y_true[valid_mask]
    
    # Calculate market implied probabilities
    market_odds = combined_data['betfairSP'].values
    market_probs = 1 / market_odds
    
    # Normalize market probabilities by race (remove overround)
    temp_df = combined_data[['Race_ID']].copy()
    temp_df['market_prob_raw'] = market_probs
    market_probs_normalized = temp_df.groupby('Race_ID')['market_prob_raw'].transform(
        lambda x: x / x.sum()
    ).values
    
    # Calculate edge (your probability - market probability)
    prediction_edge = y_pred_proba - market_probs_normalized
    
    print(f"\n=== BASIC STATISTICS ===")
    print(f"Total races analyzed: {len(np.unique(combined_data['Race_ID']))}")
    print(f"Total horses analyzed: {len(combined_data)}")
    print(f"Market odds range: {market_odds.min():.2f} to {market_odds.max():.2f}")
    print(f"Market implied prob range: {market_probs.min():.4f} to {market_probs.max():.4f}")
    print(f"Market implied prob (normalized) range: {market_probs_normalized.min():.4f} to {market_probs_normalized.max():.4f}")
    print(f"Your prediction range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
    print(f"Edge range: {prediction_edge.min():.4f} to {prediction_edge.max():.4f}")
    print(f"Mean edge: {prediction_edge.mean():.4f}")
    
    # Edge analysis by buckets
    print(f"\n=== EDGE ANALYSIS ===")
    edge_buckets = pd.cut(prediction_edge, 
                         bins=[-1, -0.15, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15, 1], 
                         labels=['Very Negative', 'Negative', 'Slightly Negative', 'Small Negative',
                                'Neutral', 'Small Positive', 'Slightly Positive', 'Positive', 'Very Positive'])
    
    edge_analysis = pd.DataFrame({
        'edge_bucket': edge_buckets,
        'actual_win': y_true,
        'your_prob': y_pred_proba,
        'market_prob': market_probs_normalized,
        'edge': prediction_edge,
        'market_odds': market_odds
    })
    
    bucket_stats = edge_analysis.groupby('edge_bucket').agg({
        'actual_win': ['count', 'sum', 'mean'],
        'edge': 'mean',
        'your_prob': 'mean',
        'market_prob': 'mean',
        'market_odds': 'mean'
    }).round(4)
    
    print("Edge Bucket Analysis:")
    print(bucket_stats)
    
    # Value betting opportunities
    print(f"\n=== VALUE BETTING OPPORTUNITIES ===")
    
    # Define different edge thresholds
    edge_thresholds = [0.02, 0.05, 0.1, 0.15]
    min_confidence_thresholds = [0.05, 0.1, 0.15, 0.2]
    
    value_opportunities = []
    
    for edge_thresh in edge_thresholds:
        for conf_thresh in min_confidence_thresholds:
            value_mask = (prediction_edge > edge_thresh) & (y_pred_proba > conf_thresh)
            
            if value_mask.sum() > 0:
                hit_rate = y_true[value_mask].mean()
                avg_edge = prediction_edge[value_mask].mean()
                avg_odds = market_odds[value_mask].mean()
                avg_your_prob = y_pred_proba[value_mask].mean()
                
                # Simulate ROI assuming flat betting
                simulated_roi = calculate_roi_simulation(market_odds[value_mask], y_true[value_mask])
                
                value_opportunities.append({
                    'edge_threshold': edge_thresh,
                    'confidence_threshold': conf_thresh,
                    'opportunities': value_mask.sum(),
                    'hit_rate': hit_rate,
                    'avg_edge': avg_edge,
                    'avg_odds': avg_odds,
                    'avg_your_prob': avg_your_prob,
                    'simulated_roi': simulated_roi
                })
    
    value_df = pd.DataFrame(value_opportunities)
    print("Value Betting Analysis (Top 10 by ROI):")
    print(value_df.nlargest(10, 'simulated_roi')[['edge_threshold', 'confidence_threshold', 
                                                   'opportunities', 'hit_rate', 'avg_edge', 'simulated_roi']])
    
    # High confidence analysis
    print(f"\n=== HIGH CONFIDENCE ANALYSIS ===")
    
    confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for conf_thresh in confidence_thresholds:
        high_conf_mask = y_pred_proba > conf_thresh
        
        if high_conf_mask.sum() > 0:
            # Your high confidence picks
            your_hit_rate = y_true[high_conf_mask].mean()
            avg_market_odds = market_odds[high_conf_mask].mean()
            
            # Market favorites in these picks
            market_fav_mask = market_probs_normalized > conf_thresh
            overlap = high_conf_mask & market_fav_mask
            
            print(f"\nConfidence > {conf_thresh}:")
            print(f"  Your picks: {high_conf_mask.sum()}")
            print(f"  Hit rate: {your_hit_rate:.3f}")
            print(f"  Avg market odds: {avg_market_odds:.2f}")
            print(f"  Overlap with market favorites: {overlap.sum()}")
            
            if overlap.sum() > 0:
                print(f"  Hit rate when agreeing with market: {y_true[overlap].mean():.3f}")
            
            # Non-market favorites in your picks
            non_fav_picks = high_conf_mask & ~market_fav_mask
            if non_fav_picks.sum() > 0:
                print(f"  Your picks that aren't market favorites: {non_fav_picks.sum()}")
                print(f"  Hit rate for non-market-favorite picks: {y_true[non_fav_picks].mean():.3f}")
                print(f"  Avg odds for non-market-favorite picks: {market_odds[non_fav_picks].mean():.2f}")
    
    # Model vs Market performance comparison
    print(f"\n=== MODEL VS MARKET PERFORMANCE ===")
    
    # AUC comparison
    market_auc = roc_auc_score(y_true, market_probs_normalized)
    model_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"Market AUC: {market_auc:.4f}")
    print(f"Your Model AUC: {model_auc:.4f}")
    print(f"AUC Improvement: {model_auc - market_auc:.4f}")
    
    # Correlation analysis
    correlation = np.corrcoef(y_pred_proba, market_probs_normalized)[0, 1]
    print(f"Correlation between your predictions and market: {correlation:.4f}")
    
    # Find best disagreement scenarios
    print(f"\n=== BEST DISAGREEMENT SCENARIOS ===")
    
    # Where you're much more confident than market
    big_positive_edge = prediction_edge > 0.1
    if big_positive_edge.sum() > 0:
        print(f"Big positive edge (>0.1) opportunities: {big_positive_edge.sum()}")
        print(f"Hit rate: {y_true[big_positive_edge].mean():.3f}")
        print(f"Average edge: {prediction_edge[big_positive_edge].mean():.4f}")
        print(f"Average odds: {market_odds[big_positive_edge].mean():.2f}")
        print(f"ROI simulation: {calculate_roi_simulation(market_odds[big_positive_edge], y_true[big_positive_edge]):.1f}%")
    
    # Where market is much more confident than you
    big_negative_edge = prediction_edge < -0.1
    if big_negative_edge.sum() > 0:
        print(f"\nBig negative edge (<-0.1) scenarios: {big_negative_edge.sum()}")
        print(f"Hit rate: {y_true[big_negative_edge].mean():.3f}")
        print(f"Market hit rate would be: {market_probs_normalized[big_negative_edge].mean():.3f}")
        print(f"Average market odds: {market_odds[big_negative_edge].mean():.2f}")
    
    return {
        'edge_analysis': edge_analysis,
        'value_opportunities': value_df,
        'market_auc': market_auc,
        'model_auc': model_auc,
        'correlation': correlation
    }


def calculate_roi_simulation(odds_array, results_array, stake=1):
    """
    Calculate ROI simulation assuming flat betting.
    
    Args:
        odds_array: Array of betting odds
        results_array: Array of results (1 for win, 0 for loss)
        stake: Stake per bet
    
    Returns:
        ROI percentage
    """
    total_staked = len(odds_array) * stake
    total_winnings = np.sum(results_array * odds_array * stake)
    profit = total_winnings - total_staked
    roi = (profit / total_staked) * 100
    return roi


def run_market_analysis():
    """
    Run the complete market analysis pipeline.
    """
    print("Loading data and running model...")
    
    # Load and preprocess data
    data = Preprocessing().check_and_preprocess()
    
    # Load original data with betfairSP
    config = Config()
    original_data = pd.read_csv(config.data_path)
    
    # Train model and get predictions
    model_components, train_data = model_run(data)
    
    # Get out-of-fold predictions and apply the same transformations as in evaluation
    scale_factor = config.softmax_scale_factor
    
    y_true = train_data['win'].values
    
    # Get raw predictions (same logic as in evaluation.py)
    if model_components['calibrator'] is not None:
        y_pred_proba_raw = model_components['calibrator'].predict_proba(model_components['oof_predictions'])[:, 1]
    else:
        if 'meta_model' in model_components and model_components['meta_model'] is not None:
            y_pred_proba_raw = model_components['meta_model'].predict_proba(model_components['oof_predictions'])[:, 1]
        else:
            y_pred_proba_raw = model_components['oof_predictions'].mean(axis=1)
    
    # Apply softmax transformation (same as in evaluation.py)
    temp_df = train_data[['Race_ID']].copy()
    temp_df['pred_raw'] = y_pred_proba_raw
    
    y_pred_proba = temp_df.groupby('Race_ID')['pred_raw'].transform(
        lambda x: np.exp(x * scale_factor) / np.exp(x * scale_factor).sum()
    ).values
    
    # Run market analysis
    results = analyze_market_comparison(train_data, original_data, y_pred_proba, y_true)
    
    return results


if __name__ == "__main__":
    results = run_market_analysis()
    print("\n" + "="*80)
    print("MARKET ANALYSIS COMPLETE")
    print("="*80) 