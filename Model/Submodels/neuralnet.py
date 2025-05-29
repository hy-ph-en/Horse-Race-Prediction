from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import pandas as pd
import numpy as np

def get_model(**kwargs):
    """
    Returns a Neural Network classifier (MLP) with manual resampling for class balancing.
    Pass hyperparameters via kwargs.
    """
    params = {
        'hidden_layer_sizes': (200, 100, 50),  # Deeper network for better pattern recognition
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 1e-6,  # Further reduced regularization for less conservative predictions
        'learning_rate_init': 0.001,
        'max_iter': 1000,  # Increased iterations
        'early_stopping': True,  # Prevent overfitting
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
        'random_state': 42,
    }
    params.update(kwargs)
    
    class BalancedMLPClassifier(MLPClassifier):
        def fit(self, X, y):
            # Convert to DataFrame for easier manipulation with proper indexing
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X)
            else:
                X_df = X.copy()
            
            # Reset index to ensure alignment
            X_df = X_df.reset_index(drop=True)
            y_series = pd.Series(y).reset_index(drop=True)
            
            # Store original column names for consistency
            self.feature_names_in_ = X_df.columns.tolist() if hasattr(X_df, 'columns') else None
            
            # Separate majority and minority classes
            majority_class = y_series.value_counts().idxmax()
            minority_class = y_series.value_counts().idxmin()
            
            majority_mask = y_series == majority_class
            minority_mask = y_series == minority_class
            
            X_majority = X_df[majority_mask]
            X_minority = X_df[minority_mask]
            y_majority = y_series[majority_mask]
            y_minority = y_series[minority_mask]
            
            # Upsample minority class (but not too aggressively)
            target_minority_size = min(len(X_majority) // 3, len(X_minority) * 4)
            
            X_minority_upsampled, y_minority_upsampled = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=target_minority_size,
                random_state=42
            )
            
            # Combine majority class with upsampled minority class
            X_balanced = pd.concat([X_majority, X_minority_upsampled], ignore_index=True)
            y_balanced = pd.concat([y_majority, y_minority_upsampled], ignore_index=True)
            
            # Shuffle the data
            indices = np.random.RandomState(42).permutation(len(X_balanced))
            X_final = X_balanced.iloc[indices].values
            y_final = y_balanced.iloc[indices].values
            
            return super().fit(X_final, y_final)
        
        def predict_proba(self, X):
            # Convert DataFrame to arrays to maintain consistency
            if hasattr(X, 'values'):
                X = X.values
            return super().predict_proba(X)
    
    return BalancedMLPClassifier(**params)