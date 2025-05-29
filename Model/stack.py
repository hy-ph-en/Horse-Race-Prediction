import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from configuration import Config

# import base model getters
from .Submodels.light_gmb import get_model as get_lgbm
from .Submodels.xgboost import get_model as get_xgb
from .Submodels.neuralnet import get_model as get_nn
from .Submodels.logistic_regression import get_model as get_lr

# meta-learner
from xgboost import XGBClassifier


def prepare_features(df, feature_cols):
    """
    Impute and scale features while preserving feature names.
    """
    # Extract features as DataFrame to preserve names
    feature_df = df[feature_cols].copy()
    
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    
    # Fit and transform while preserving structure
    X_imputed = imputer.fit_transform(feature_df)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Convert back to DataFrame with original column names
    X_df = pd.DataFrame(X_scaled, columns=feature_cols, index=feature_df.index)
    
    return X_df, imputer, scaler


def stacking_train_predict(train_df, test_df, feature_cols, target_col, group_col):
    """
    1) Trains base models with out-of-fold predictions
    2) Trains an XGBoost meta-model on base predictions
    3) Outputs final test-set probabilities
    4) Returns trained model components for evaluation
    """
    # Prepare feature matrix (returns DataFrame now)
    X_all_df, imputer, scaler = prepare_features(train_df, feature_cols)
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
    
    # Balanced approach: calibration + moderate softmax for optimal Log Loss/Brier Score
    calibrator = CalibratedClassifierCV(meta_model, method='isotonic', cv=3)
    calibrator.fit(oof_preds, y_all)
    
    # Get calibrated predictions for better probabilistic accuracy
    final_test_raw = calibrator.predict_proba(test_feats)[:, 1]
    
    # Optional: Use raw predictions (commented out for better calibration)
    # meta_model.fit(oof_preds, y_all)
    # final_test_raw = meta_model.predict_proba(test_feats)[:, 1]
    
    print(f"Calibrated meta-model predictions:")
    print(f"Range: {final_test_raw.min():.4f} to {final_test_raw.max():.4f}")
    print(f"Mean: {final_test_raw.mean():.4f}")
    
    # Final predictions
    test_df['pred_raw'] = final_test_raw
    
    # Balanced softmax: more aggressive scale factor to overcome conservative calibration
    test_df['Predicted_Probability'] = test_df.groupby(group_col)['pred_raw'].transform(
        lambda x: np.exp(x * 6) / np.exp(x * 6).sum()  # Scale factor 6: more aggressive to generate some winner predictions
    )
    
    print(f"After normalization:")
    print(f"Final prediction range: {test_df['Predicted_Probability'].min():.4f} to {test_df['Predicted_Probability'].max():.4f}")
    print(f"Final prediction mean: {test_df['Predicted_Probability'].mean():.4f}")
    
    # Option 1: Weighted normalization (preserves more original signal) - now commented out
    # epsilon = 1e-8
    # test_df['Predicted_Probability'] = test_df.groupby(group_col)['pred_raw'].transform(
    #     lambda x: (x + epsilon) / (x.sum() + len(x) * epsilon)
    # )

    # Prepare model components for return
    model_components = {
        'calibrator': calibrator,
        'meta_model': meta_model,
        'trained_base_models': trained_base_models,
        'imputer': imputer,
        'scaler': scaler,
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
