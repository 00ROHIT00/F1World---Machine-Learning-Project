import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create sample data
np.random.seed(42)
n_samples = 100

# Generate realistic F1 pit stop data
tires_changed = np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.1, 0.1, 0.7])  # Most common is 4 tires
pit_stop_time = []  # Pit stop time in seconds

for tires in tires_changed:
    # Base time for stopping and starting: 1.0-1.5 seconds
    base_time = 1.0 + np.random.uniform(0, 0.5)
    
    # Time per tire change: 0.5-0.8 seconds per tire
    tire_change_time = sum(np.random.uniform(0.5, 0.8) for _ in range(tires))
    
    # Add some random variation (crew performance, conditions)
    random_factor = np.random.normal(0, 0.2)  # Small random variation
    
    # Calculate total time
    total_time = base_time + tire_change_time + random_factor
    
    # Ensure minimum pit stop time is 1.9 seconds (F1 record is ~1.91s)
    total_time = max(1.9, total_time)
    
    pit_stop_time.append(total_time)

# Convert to numpy array
pit_stop_time = np.array(pit_stop_time)

# Create DataFrame with exact column name matching views.py
data = pd.DataFrame({
    'Number of Tires Changed': tires_changed,
    'Pit Stop Time (s)': pit_stop_time
})

# Save the dataset
data.to_excel(os.path.join(script_dir, 'SLR.xlsx'), index=False)

# Prepare features and target
X = data[['Number of Tires Changed']]  # Keep the exact column name
y = data['Pit Stop Time (s)']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Calculate R-squared score
r2_score = model.score(X_scaled, y)

# Create models directory if it doesn't exist
models_dir = os.path.join(script_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# Save model and scaler
joblib.dump(model, os.path.join(models_dir, 'slr_model.joblib'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))

# Save model metrics
metrics = {
    'r2_score': r2_score,
    'coefficient': float(model.coef_[0]),
    'intercept': float(model.intercept_)
}
with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f)

print("Model training completed and saved successfully!")
print("\nModel details:")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R² Score: {r2_score:.4f}")

# Print some example predictions
print("\nExample predictions:")
test_tires = np.array([1, 2, 3, 4]).reshape(-1, 1)
test_tires_scaled = scaler.transform(test_tires)
predictions = model.predict(test_tires_scaled)

for tires, pred in zip(test_tires.flatten(), predictions):
    print(f"Tires changed: {int(tires)} -> Pit stop time: {pred:.2f}s") 