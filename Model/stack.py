import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from configuration import Config

# import base model getters
from .Submodels.light_gmb import get_model as get_lgbm
from .Submodels.xgboost import get_model as get_xgb
from .Submodels.neuralnet import get_model as get_nn
from .Submodels.logistic_regression import get_model as get_lr

# meta-learner
from xgboost import XGBClassifier

# Import diagnostics
from Evaluation.diagnostics import run_detailed_diagnostics

# Import feature preparation from Data_Processing
from Data_Processing.feature_preparation import prepare_features


def stacking_train_predict(train_df, test_df, feature_cols, target_col, group_col):
    """
    1) Trains base models with out-of-fold predictions
    2) Trains an XGBoost meta-model on base predictions
    3) Outputs final test-set probabilities
    4) Returns trained model components for evaluation
    """
    # Load configuration
    config = Config()
    
    # Prepare feature matrix (returns DataFrame now)
    X_all_df, preprocessors, _ = prepare_features(train_df, feature_cols)
    y_all = train_df[target_col].values

    # Placeholder for OOF and test preds
    oof_preds = np.zeros((len(train_df), 4))
    test_feats = np.zeros((len(test_df), 4))

    base_getters = [get_lgbm, get_xgb, get_nn, get_lr]
    trained_base_models = []

    # Prepare test features once (returns DataFrame)
    X_test_df, _, _ = prepare_features(test_df, feature_cols)

    # GroupKFold by races
    gkf = GroupKFold(n_splits=5)
    print(f"Training base models with {gkf.n_splits} folds...")
    print(f"Class distribution: {np.bincount(y_all)} (0: no win, 1: win)")
    print(f"Positive class ratio: {np.mean(y_all):.3f}")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all_df, y_all, groups=train_df[group_col])):
        X_tr_df = X_all_df.iloc[train_idx]
        X_val_df = X_all_df.iloc[val_idx]
        y_tr = y_all[train_idx]
        y_val = y_all[val_idx]
        
        print(f"Fold {fold + 1}: Train samples: {len(y_tr)}, Val samples: {len(y_val)}")
        print(f"  Train win rate: {np.mean(y_tr):.3f}, Val win rate: {np.mean(y_val):.3f}")
        
        fold_models = []
        for m_idx, get_model in enumerate(base_getters):
            model = get_model()
            model.fit(X_tr_df, y_tr)
            fold_models.append(model)
            # predict_proba class 1 (win)
            val_preds = model.predict_proba(X_val_df)[:, 1]
            oof_preds[val_idx, m_idx] = val_preds
            print(f"    Model {m_idx} - Val pred range: {val_preds.min():.3f} to {val_preds.max():.3f}, Mean: {val_preds.mean():.3f}")
            print(f"    Model {m_idx} - Preds >0.5: {np.sum(val_preds > 0.5)}/{len(val_preds)} ({100*np.sum(val_preds > 0.5)/len(val_preds):.1f}%)")
            # accumulate test preds
            test_feats[:, m_idx] += model.predict_proba(X_test_df)[:, 1] / gkf.n_splits
        
        trained_base_models.append(fold_models)

    # Train meta-model on OOF
    print(f"\n=== BASE MODEL ANALYSIS ===")
    print(f"OOF predictions shape: {oof_preds.shape}")
    model_names = ['LightGBM', 'XGBoost', 'NeuralNet', 'LogisticReg']
    
    for i in range(oof_preds.shape[1]):
        base_preds = oof_preds[:, i]
        print(f"{model_names[i]}:")
        print(f"  Range: {base_preds.min():.3f} to {base_preds.max():.3f}")
        print(f"  Mean: {base_preds.mean():.3f}")
        print(f"  Preds >0.5: {np.sum(base_preds > 0.5)}/{len(base_preds)} ({100*np.sum(base_preds > 0.5)/len(base_preds):.1f}%)")
        print(f"  Preds >0.2: {np.sum(base_preds > 0.2)}/{len(base_preds)} ({100*np.sum(base_preds > 0.2)/len(base_preds):.1f}%)")
    
    # Simple ensemble for comparison
    simple_avg = oof_preds.mean(axis=1)
    print(f"\nSimple Average Ensemble:")
    print(f"  Range: {simple_avg.min():.3f} to {simple_avg.max():.3f}")
    print(f"  Mean: {simple_avg.mean():.3f}")
    print(f"  Preds >0.5: {np.sum(simple_avg > 0.5)}/{len(simple_avg)} ({100*np.sum(simple_avg > 0.5)/len(simple_avg):.1f}%)")
    
    print(f"\n=== TRAINING META-MODEL ===")
    
    # Initialize meta-model
    meta_model = XGBClassifier(
        n_estimators=100,  # Increased from 50
        learning_rate = 0.05,  # Reduced for more careful learning
        max_depth=5,  # Increased depth for better pattern recognition
        min_child_weight=1,  # Reduced for more granular splits
        subsample=0.8,  
        colsample_bytree=0.8,  
        reg_alpha=0.01,  # Reduced regularization
        reg_lambda=0.01,  # Reduced regularization
        scale_pos_weight=4.0,  # Reduced from 8.5 - less aggressive class balancing
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0  
    )
    
    # Fit the meta_model on out-of-fold predictions
    meta_model.fit(oof_preds, y_all)
    
    # Configurable calibration - use config settings
    calibrator = None
    if config.use_meta_calibration:
        print(f"Applying {config.calibration_method} calibration with {config.calibration_cv_folds} CV folds...")
        calibrator = CalibratedClassifierCV(
            meta_model, 
            method=config.calibration_method, 
            cv=config.calibration_cv_folds
        )
        calibrator.fit(oof_preds, y_all)
        final_test_raw = calibrator.predict_proba(test_feats)[:, 1]
        print(f"Calibrated meta-model predictions:")
    else:
        # Use raw predictions (current approach)
        final_test_raw = meta_model.predict_proba(test_feats)[:, 1]
        print(f"Raw meta-model predictions:")
    
    print(f"Range: {final_test_raw.min():.4f} to {final_test_raw.max():.4f}")
    print(f"Mean: {final_test_raw.mean():.4f}")
    
    # Final predictions
    test_df['pred_raw'] = final_test_raw
    
    # Run detailed diagnostics if configured
    if config.show_detailed_diagnostics:
        run_detailed_diagnostics(train_df, test_df, meta_model, oof_preds, group_col)
    
    # Use configurable scale factor for final predictions
    if config.use_alternative_scaling:
        # Option: Weighted normalization (preserves more original signal)
        print(f"\nUsing alternative weighted normalization with epsilon={config.epsilon_scaling}")
        test_df['Predicted_Probability'] = test_df.groupby(group_col)['pred_raw'].transform(
            lambda x: (x + config.epsilon_scaling) / (x.sum() + len(x) * config.epsilon_scaling)
        )
    else:
        # Standard softmax normalization with configurable scale factor
        print(f"\nUsing softmax normalization with scale factor={config.softmax_scale_factor}")
        test_df['Predicted_Probability'] = test_df.groupby(group_col)['pred_raw'].transform(
            lambda x: np.exp(x * config.softmax_scale_factor) / np.exp(x * config.softmax_scale_factor).sum()
        )
    
    print(f"After normalization:")
    print(f"Final prediction range: {test_df['Predicted_Probability'].min():.4f} to {test_df['Predicted_Probability'].max():.4f}")
    print(f"Final prediction mean: {test_df['Predicted_Probability'].mean():.4f}")

    # Prepare model components for return
    model_components = {
        'calibrator': calibrator,
        'meta_model': meta_model,
        'trained_base_models': trained_base_models,
        'preprocessors': preprocessors,
        'oof_predictions': oof_preds,
        'feature_cols': feature_cols
    }

    return test_df[[group_col, 'Horse', 'Predicted_Probability']], model_components


def model_run(data=None):
    """
    Main model training function.
    
    Args:
        data: Optional preprocessed data. If None, loads from config paths.
    
    Returns:
        tuple: (model_components, train_data) for evaluation
    """
    #Class Loading
    config = Config()

    if data is None:
        #Loading Data
        train = pd.read_csv(config.data_path)
        test  = pd.read_csv(config.test_data_path)
    else:
        train, test = data

    # Create binary target
    train['win'] = (train['Position'] == 1).astype(int)

    # Define your feature columns (e.g., pre-engineered race-relative features)
    feature_cols = config.feature_cols

    submission, model_components = stacking_train_predict(
        train_df=train,
        test_df=test,
        feature_cols=feature_cols,
        target_col='win',
        group_col='Race_ID'
    )

    #Sort by Race_ID
    submission = submission.sort_values('Race_ID').copy()
    
    submission.to_csv('submission.csv', index=False)
    
    return model_components, train