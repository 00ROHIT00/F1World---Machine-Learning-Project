import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to reach F1World directory

try:
    # Load the dataset
    data_path = os.path.join(project_dir, 'main', 'data', 'LR.xlsx')
    print(f"Loading data from: {data_path}")
    data = pd.read_excel(data_path)
    
    print("\nDataFrame columns:")
    print(data.columns.tolist())
    print("\nFirst few rows:")
    print(data.head())
    
    # Prepare features and target
    features = ['Qualifying Position', 'Past Race Performance', 'Track Characteristics']
    X = data[features]
    y = data['Podium Finish']  # 1 for podium, 0 for no podium

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    # Make predictions on training data
    y_pred = model.predict(X_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    classification_metrics = classification_report(y, y_pred, output_dict=True)

    # Create models directory if it doesn't exist
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save model and scaler
    joblib.dump(model, os.path.join(models_dir, 'lr_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))

    # Save model metrics
    metrics = {
        'accuracy': accuracy,
        'precision': classification_metrics['1']['precision'],  # Precision for podium finish
        'recall': classification_metrics['1']['recall'],  # Recall for podium finish
        'f1_score': classification_metrics['1']['f1-score'],  # F1 score for podium finish
        'coefficients': {feature: float(coef) for feature, coef in zip(features, model.coef_[0])},
        'intercept': float(model.intercept_[0])
    }
    
    with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print("\nModel training completed and saved successfully!")
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Podium): {metrics['precision']:.4f}")
    print(f"Recall (Podium): {metrics['recall']:.4f}")
    print(f"F1 Score (Podium): {metrics['f1_score']:.4f}")
    
    print("\nFeature Importance:")
    for feature, coefficient in metrics['coefficients'].items():
        print(f"{feature}: {coefficient:.4f}")

    # Print some example predictions
    print("\nExample predictions:")
    example_data = pd.DataFrame({
        'Qualifying Position': [1, 5, 10],
        'Past Race Performance': [1, 0.8, 0.5],
        'Track Characteristics': [0.9, 0.7, 0.3]
    })
    example_scaled = scaler.transform(example_data)
    example_pred = model.predict(example_scaled)
    example_prob = model.predict_proba(example_scaled)

    for i, (pred, prob) in enumerate(zip(example_pred, example_prob)):
        position = example_data['Qualifying Position'].iloc[i]
        performance = example_data['Past Race Performance'].iloc[i]
        track = example_data['Track Characteristics'].iloc[i]
        print(f"\nDriver with:")
        print(f"  Qualifying Position: {position}")
        print(f"  Past Race Performance: {performance:.1f}")
        print(f"  Track Characteristics: {track:.1f}")
        print(f"Prediction: {'Podium' if pred == 1 else 'No Podium'}")
        print(f"Probability of Podium: {prob[1]:.2%}")

except Exception as e:
    print(f"Error during model training: {str(e)}")
    if 'data' in locals():
        print("\nAvailable columns in the dataset:")
        print(data.columns.tolist()) 