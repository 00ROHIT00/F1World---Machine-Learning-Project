import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

# Create sample data
np.random.seed(42)
n_samples = 100

# Generate realistic F1 data
qualifying_positions = np.random.randint(1, 21, n_samples)
overtakes = np.random.randint(0, 15, n_samples)
pit_stops = np.random.randint(1, 4, n_samples)
fastest_lap = np.random.binomial(1, 0.05, n_samples)  # 5% chance of fastest lap

# Calculate points based on a realistic formula
points = np.zeros(n_samples)
for i in range(n_samples):
    # Base points based on qualifying (better qualifying generally means more points)
    base_points = max(25 * (1 - (qualifying_positions[i] - 1) / 20), 0)
    
    # Add points for overtakes
    overtake_bonus = overtakes[i] * 0.5
    
    # Subtract points for extra pit stops
    pit_penalty = (pit_stops[i] - 1) * 2
    
    # Add point for fastest lap (only if in top 10)
    fastest_lap_points = fastest_lap[i] * (1 if base_points > 0 else 0)
    
    # Calculate total points with some random variation
    points[i] = max(0, base_points + overtake_bonus - pit_penalty + fastest_lap_points + np.random.normal(0, 2))

# Create DataFrame
data = pd.DataFrame({
    'Qualifying Position': qualifying_positions,
    'Number of Overtakes': overtakes,
    'Pit Stops': pit_stops,
    'Fastest Lap Bonus': fastest_lap,
    'Points Scored': points
})

# Save the dataset
data.to_excel('main/MLR/MLR.xlsx', index=False)

# Prepare features and target
X = data[['Qualifying Position', 'Number of Overtakes', 'Pit Stops', 'Fastest Lap Bonus']]
y = data['Points Scored']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Calculate R-squared score
r2_score = model.score(X_scaled, y)

# Create models directory if it doesn't exist
os.makedirs('main/MLR/models', exist_ok=True)

# Save model and scaler
joblib.dump(model, 'main/MLR/models/mlr_model.joblib')
joblib.dump(scaler, 'main/MLR/models/scaler.joblib')

# Save model metrics
metrics = {
    'r2_score': r2_score,
    'coefficients': {feature: float(coef) for feature, coef in zip(X.columns, model.coef_)},
    'intercept': float(model.intercept_)
}
with open('main/MLR/models/model_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Model training completed and saved successfully!")
print("\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"\nModel R-squared score: {r2_score:.4f}") 