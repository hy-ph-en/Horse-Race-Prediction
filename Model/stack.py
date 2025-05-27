import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from configuration import Config

# import base model getters
from Submodels.light_gmb import get_model as get_lgbm
from Submodels.xgboost import get_model as get_xgb
from Submodels.neuralnet import get_model as get_nn
from Submodels.logistic_regression import get_model as get_lr

# meta-learner
from xgboost import XGBClassifier


def prepare_features(df, feature_cols):
    """
    Impute and scale features.
    """
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    X = imputer.fit_transform(df[feature_cols])
    X = scaler.fit_transform(X)
    return X, imputer, scaler


def stacking_train_predict(train_df, test_df, feature_cols, target_col, group_col):
    """
    1) Trains base models with out-of-fold predictions
    2) Trains an XGBoost meta-model on base predictions
    3) Outputs final test-set probabilities
    4) Returns trained model components for evaluation
    """
    # Prepare feature matrix
    X_all, imputer, scaler = prepare_features(train_df, feature_cols)
    y_all = train_df[target_col].values

    # Placeholder for OOF and test preds
    oof_preds = np.zeros((len(train_df), 4))
    test_feats = np.zeros((len(test_df), 4))

    base_getters = [get_lgbm, get_xgb, get_nn, get_lr]
    trained_base_models = []

    # GroupKFold by races
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=train_df[group_col])):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_val = X_all[val_idx]
        
        fold_models = []
        for m_idx, get_model in enumerate(base_getters):
            model = get_model()
            model.fit(X_tr, y_tr)
            fold_models.append(model)
            # predict_proba class 1 (win)
            oof_preds[val_idx, m_idx] = model.predict_proba(X_val)[:, 1]
            # accumulate test preds
            X_test, _, _ = prepare_features(test_df, feature_cols)
            test_feats[:, m_idx] += model.predict_proba(X_test)[:, 1] / gkf.n_splits
        
        trained_base_models.append(fold_models)

    # Train meta-model on OOF
    meta_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    # Optionally calibrate
    calibrator = CalibratedClassifierCV(meta_model, method='sigmoid', cv=3)
    calibrator.fit(oof_preds, y_all)

    # Final predictions
    final_test_raw = calibrator.predict_proba(test_feats)[:, 1]

    # Normalize per race
    test_df['pred_raw'] = final_test_raw
    test_df['Predicted_Probability'] = test_df.groupby(group_col)['pred_raw'].transform(lambda x: x / x.sum())

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
    
    submission.to_csv('submission.csv', index=False)
    
    return model_components, train
