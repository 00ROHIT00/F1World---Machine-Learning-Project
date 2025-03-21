import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    # Read the Excel file
    df = pd.read_excel('../data/SLR.xlsx')
    
    # Basic data info
    print("Dataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Test the preprocessing
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print("\nPreprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}") 