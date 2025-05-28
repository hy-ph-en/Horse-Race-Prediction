import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Import base model getters
from Model.Submodels.light_gmb import get_model as get_lgbm
from Model.Submodels.xgboost import get_model as get_xgb
from Model.Submodels.neuralnet import get_model as get_nn
from Model.Submodels.logistic_regression import get_model as get_lr

# Meta-learner
from xgboost import XGBClassifier

from Data_Processing.preprocessing import Preprocessing
from configuration import Config


class HyperparameterOptimizer:
    def __init__(self, train_data, feature_cols, target_col='win', group_col='Race_ID', n_trials=100):
        self.train_data = train_data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.group_col = group_col
        self.n_trials = n_trials
        
        # Prepare features
        self.X, self.imputer, self.scaler = self.prepare_features(train_data, feature_cols)
        self.y = train_data[target_col].values
        self.groups = train_data[group_col].values
        
        # Cross-validation strategy
        self.cv = GroupKFold(n_splits=3)  # Reduced for faster optimization
        
        # Best parameters storage
        self.best_params = {}
        
    def prepare_features(self, df, feature_cols):
        """Prepare features with imputation and scaling"""
        feature_df = df[feature_cols].copy()
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_imputed = imputer.fit_transform(feature_df)
        X_scaled = scaler.fit_transform(X_imputed)
        
        X_df = pd.DataFrame(X_scaled, columns=feature_cols, index=feature_df.index)
        
        return X_df, imputer, scaler
    
    def objective_lgbm(self, trial):
        """Objective function for LightGBM optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
        }
        
        model = get_lgbm(**params)
        
        scores = []
        for train_idx, val_idx in self.cv.split(self.X, self.y, self.groups):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def objective_xgb(self, trial):
        """Objective function for XGBoost optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
        }
        
        model = get_xgb(**params)
        
        scores = []
        for train_idx, val_idx in self.cv.split(self.X, self.y, self.groups):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def objective_nn(self, trial):
        """Objective function for Neural Network optimization"""
        hidden_size_1 = trial.suggest_int('hidden_size_1', 50, 200)
        hidden_size_2 = trial.suggest_int('hidden_size_2', 25, 100)
        
        params = {
            'hidden_layer_sizes': (hidden_size_1, hidden_size_2),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 500)
        }
        
        model = get_nn(**params)
        
        scores = []
        for train_idx, val_idx in self.cv.split(self.X, self.y, self.groups):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def objective_lr(self, trial):
        """Objective function for Logistic Regression optimization"""
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
        }
        
        # Handle solver compatibility
        if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
            params['solver'] = 'liblinear'
        
        model = get_lr(**params)
        
        scores = []
        for train_idx, val_idx in self.cv.split(self.X, self.y, self.groups):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def get_stacking_oof_predictions(self, best_params):
        """Generate OOF predictions using optimized base models"""
        oof_preds = np.zeros((len(self.train_data), 4))
        
        base_getters = [get_lgbm, get_xgb, get_nn, get_lr]
        model_names = ['lgbm', 'xgb', 'nn', 'lr']
        
        gkf = GroupKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(self.X, self.y, self.groups)):
            X_tr = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_tr = self.y[train_idx]
            
            for m_idx, (get_model, model_name) in enumerate(zip(base_getters, model_names)):
                model = get_model(**best_params[model_name])
                model.fit(X_tr, y_tr)
                oof_preds[val_idx, m_idx] = model.predict_proba(X_val)[:, 1]
        
        return oof_preds
    
    def objective_meta(self, trial):
        """Objective function for meta model optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 30, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
        }
        
        # Get OOF predictions from optimized base models
        oof_preds = self.get_stacking_oof_predictions(self.best_params)
        
        meta_model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
            **params
        )
        
        scores = []
        for train_idx, val_idx in self.cv.split(oof_preds, self.y, self.groups):
            X_meta_train, X_meta_val = oof_preds[train_idx], oof_preds[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            meta_model.fit(X_meta_train, y_train)
            y_pred = meta_model.predict_proba(X_meta_val)[:, 1]
            score = log_loss(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def optimize_all_models(self):
        """Optimize all models sequentially"""
        print("Starting hyperparameter optimization...")
        
        # Optimize base models
        models_to_optimize = [
            ('lgbm', self.objective_lgbm),
            ('xgb', self.objective_xgb),
            ('nn', self.objective_nn),
            ('lr', self.objective_lr)
        ]
        
        for model_name, objective_func in models_to_optimize:
            print(f"\nOptimizing {model_name.upper()} model...")
            study = optuna.create_study(direction='minimize', study_name=f'{model_name}_optimization')
            study.optimize(objective_func, n_trials=self.n_trials//4)  # Distribute trials
            
            self.best_params[model_name] = study.best_params
            print(f"Best {model_name.upper()} score: {study.best_value:.4f}")
            print(f"Best {model_name.upper()} params: {study.best_params}")
        
        # Optimize meta model
        print(f"\nOptimizing META model...")
        meta_study = optuna.create_study(direction='minimize', study_name='meta_optimization')
        meta_study.optimize(self.objective_meta, n_trials=self.n_trials//4)
        
        self.best_params['meta'] = meta_study.best_params
        print(f"Best META score: {meta_study.best_value:.4f}")
        print(f"Best META params: {meta_study.best_params}")
        
        return self.best_params
    
    def save_best_params(self, filename='best_hyperparameters.txt'):
        """Save the best parameters to a file"""
        with open(filename, 'w') as f:
            f.write("BEST HYPERPARAMETERS\n")
            f.write("="*50 + "\n\n")
            
            for model_name, params in self.best_params.items():
                f.write(f"{model_name.upper()} MODEL:\n")
                f.write("-" * 20 + "\n")
                for param, value in params.items():
                    f.write(f"{param}: {value}\n")
                f.write("\n")
        
        print(f"Best parameters saved to {filename}")


def run_hyperparameter_optimization():
    """Main function to run hyperparameter optimization"""
    print("Loading and preprocessing data...")
    
    # Load data
    data = Preprocessing().check_and_preprocess()
    train_data, _ = data
    
    # Create binary target
    train_data['win'] = (train_data['Position'] == 1).astype(int)
    
    # Get feature columns from config
    config = Config()
    feature_cols = config.feature_cols
    
    print(f"Using {len(feature_cols)} features for optimization")
    print(f"Training data shape: {train_data.shape}")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        train_data=train_data,
        feature_cols=feature_cols,
        target_col='win',
        group_col='Race_ID',
        n_trials=80  # Adjust based on computational budget
    )
    
    # Run optimization
    best_params = optimizer.optimize_all_models()
    
    # Save results
    optimizer.save_best_params()
    
    return best_params


if __name__ == "__main__":
    best_params = run_hyperparameter_optimization() 