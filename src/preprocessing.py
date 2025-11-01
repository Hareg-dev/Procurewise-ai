"""
Data preprocessing module for procurement risk prediction.
Handles data cleaning, feature engineering, and preparation for ML training.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import config


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset."""
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != config.TARGET_COLUMN:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for better prediction."""
    df_eng = df.copy()
    
    # Financial risk indicators
    df_eng['revenue_spend_ratio'] = df_eng['annual_spend'] / (df_eng['annual_revenue'] + 1)
    df_eng['contract_revenue_ratio'] = df_eng['contract_value'] / (df_eng['annual_revenue'] + 1)
    
    # Performance indicators
    df_eng['payment_risk_score'] = df_eng['avg_payment_delay_days'] * df_eng['late_ratio']
    df_eng['delivery_risk_score'] = (100 - df_eng['delivery_score']) * df_eng['dispute_rate']
    
    # Relationship strength
    df_eng['relationship_strength'] = df_eng['relationship_years'] * df_eng['financial_stability_index']
    
    return df_eng


def encode_categorical_features(df: pd.DataFrame, fit_encoders: bool = True) -> tuple:
    """Encode categorical features using label encoding."""
    df_encoded = df.copy()
    encoders = {}
    
    for col in config.CATEGORICAL_FEATURES:
        if col in df_encoded.columns:
            if fit_encoders:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                encoders[col] = encoder
            else:
                # Load existing encoder
                encoder_path = config.MODELS_DIR / f"{col}_encoder.pkl"
                if encoder_path.exists():
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                    df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                    encoders[col] = encoder
    
    return df_encoded, encoders


def encode_target_variable(y: pd.Series, fit_encoder: bool = True) -> tuple:
    """Encode target variable using label encoding."""
    if fit_encoder:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        # Save encoder
        with open(config.ENCODER_PATH, 'wb') as f:
            pickle.dump(encoder, f)
    else:
        # Load existing encoder
        with open(config.ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        y_encoded = encoder.transform(y)
    
    return y_encoded, encoder


def scale_features(X: pd.DataFrame, fit_scaler: bool = True) -> tuple:
    """Scale numerical features using StandardScaler."""
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # Load existing scaler
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def prepare_features_target(df: pd.DataFrame) -> tuple:
    """Separate features and target variable."""
    # Select feature columns
    feature_cols = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    
    # Add engineered features
    engineered_cols = ['revenue_spend_ratio', 'contract_revenue_ratio', 
                      'payment_risk_score', 'delivery_risk_score', 'relationship_strength']
    
    available_features = [col for col in feature_cols + engineered_cols if col in df.columns]
    
    X = df[available_features]
    y = df[config.TARGET_COLUMN]
    
    return X, y


def preprocess_data(data_path: str = None) -> dict:
    """
    Complete preprocessing pipeline.
    Returns preprocessed data and fitted transformers.
    """
    # Load data
    if data_path is None:
        data_path = config.MERGED_TRAINING_DATA
    
    df = load_data(data_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_eng = engineer_features(df_clean)
    
    # Prepare features and target
    X, y = prepare_features_target(df_eng)
    
    # Encode target variable
    y_encoded, target_encoder = encode_target_variable(y, fit_encoder=True)
    
    # Encode categorical features
    X_encoded, encoders = encode_categorical_features(X, fit_encoders=True)
    
    # Scale features
    X_scaled, scaler = scale_features(X_encoded, fit_scaler=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Save transformers
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    for col, encoder in encoders.items():
        encoder_path = config.MODELS_DIR / f"{col}_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler,
        'encoders': encoders,
        'target_encoder': target_encoder
    }


def preprocess_single_input(input_data: dict) -> np.ndarray:
    """Preprocess single input for prediction."""
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Engineer features
    df_eng = engineer_features(df)
    
    # Select feature columns only (no target)
    feature_cols = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    engineered_cols = ['revenue_spend_ratio', 'contract_revenue_ratio', 
                      'payment_risk_score', 'delivery_risk_score', 'relationship_strength']
    
    available_features = [col for col in feature_cols + engineered_cols if col in df_eng.columns]
    X = df_eng[available_features]
    
    # Encode categorical features
    X_encoded, _ = encode_categorical_features(X, fit_encoders=False)
    
    # Scale features
    X_scaled, _ = scale_features(X_encoded, fit_scaler=False)
    
    return X_scaled