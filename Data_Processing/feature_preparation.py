import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_features(df, feature_cols):
    """
    Impute and scale features while preserving feature names.
    Handles both numeric and categorical features appropriately.
    
    Args:
        df: DataFrame containing the features
        feature_cols: List of feature column names to process
    
    Returns:
        tuple: (processed_features_df, preprocessors, None)
            - processed_features_df: DataFrame with processed features
            - preprocessors: Dictionary containing fitted preprocessing objects
            - None: Third return value for compatibility
    """
    # Extract features as DataFrame to preserve names
    feature_df = df[feature_cols].copy()
    
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if col in feature_df.columns:
            if feature_df[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
    
    print(f"Preparing {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features...")
    
    # Process numeric features
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy='median')
        numeric_scaler = StandardScaler()
        
        X_numeric = feature_df[numeric_cols]
        X_numeric_imputed = numeric_imputer.fit_transform(X_numeric)
        X_numeric_scaled = numeric_scaler.fit_transform(X_numeric_imputed)
        
        # Convert back to DataFrame
        numeric_df = pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=feature_df.index)
    else:
        numeric_df = pd.DataFrame(index=feature_df.index)
        numeric_imputer = None
        numeric_scaler = None
    
    # Process categorical features
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        label_encoders = {}
        
        categorical_df = pd.DataFrame(index=feature_df.index)
        
        for col in categorical_cols:
            # Impute missing values
            col_data = feature_df[[col]]
            col_imputed = categorical_imputer.fit_transform(col_data)
            
            # Label encode
            le = LabelEncoder()
            col_encoded = le.fit_transform(col_imputed.ravel())
            
            # Store encoder for later use
            label_encoders[col] = le
            
            # Add to dataframe
            categorical_df[col] = col_encoded
    else:
        categorical_df = pd.DataFrame(index=feature_df.index)
        categorical_imputer = None
        label_encoders = {}
    
    # Combine numeric and categorical features
    X_df = pd.concat([numeric_df, categorical_df], axis=1)
    
    # Ensure column order matches original feature_cols
    X_df = X_df[feature_cols]
    
    # Package preprocessing objects
    preprocessors = {
        'numeric_imputer': numeric_imputer,
        'numeric_scaler': numeric_scaler,
        'categorical_imputer': categorical_imputer,
        'label_encoders': label_encoders,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    
    return X_df, preprocessors, None  # Third return value for compatibility 