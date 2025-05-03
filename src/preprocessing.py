import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_adult_data(input_path, test_size=0.2, random_state=42):
    """
    Preprocess the Adult Income dataset for machine learning
    
    Args:
        input_path: Path to the raw data file
        test_size: Size of the test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, encoders, scaler
    """
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race',
        'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
        'native_country', 'income'
    ]
    
    df = pd.read_csv(input_path, names=column_names, skipinitialspace=True, header=None)
    
    # Clean the data
    df = df.replace(' ?', np.nan)
    df = df.dropna()    
    X = df.drop('income', axis=1)
    
    df['income'] = df['income'].str.strip()
    y = df['income'].apply(lambda x: 1 if x.endswith('>50K') else 0)
    print(f"Class distribution: {np.bincount(y)}")    
    # Identify categorical and numerical features
    cat_features = X.select_dtypes(include=['object']).columns
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Encode categorical features
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    
    # Split into train and test sets with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set class distribution: {np.bincount(y_train)}")
    print(f"Test set class distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, list(X.columns), encoders, scaler

def transform_new_data(X_new, encoders, scaler, feature_names):
    """
    Transform new data using the same preprocessing steps as training data
    
    Args:
        X_new: New data to transform
        encoders: Dictionary of label encoders for categorical features
        scaler: StandardScaler for numerical features
        feature_names: List of feature names
        
    Returns:
        Transformed data
    """
    X_copy = X_new.copy()
    
    # Encode categorical features
    cat_features = [col for col in X_copy.columns if col in encoders]
    for col in cat_features:
        X_copy[col] = X_copy[col].map(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
        X_copy[col] = encoders[col].transform(X_copy[col])
    
    num_features = [col for col in X_copy.columns if col not in encoders]
    X_copy[num_features] = scaler.transform(X_copy[num_features])
    X_transformed = pd.DataFrame(X_copy, columns=feature_names)
    
    return X_transformed

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, encoders, scaler = preprocess_adult_data("data/adult.data")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
