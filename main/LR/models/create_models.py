import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create a simple dummy dataset for initialization
X_dummy = np.random.rand(100, 3)  # 100 samples, 3 features
y_dummy = np.random.randint(0, 2, 100)  # Binary classification

# Create and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummy)

# Create and fit the model
model = LogisticRegression()
model.fit(X_scaled, y_dummy)

# Save the model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'lr_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Model files created successfully in {current_dir}!")
print(f"Model path: {model_path}")
print(f"Scaler path: {scaler_path}") 