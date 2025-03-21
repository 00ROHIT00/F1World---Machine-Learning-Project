from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

class SLRModel:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the model with the given data"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training results
        y_pred = self.model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        print("\nTraining Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Coefficients: {self.model.coef_}")
        print(f"Intercept: {self.model.intercept_}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise Exception("Model needs to be trained first!")
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nTest Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return mse, r2
    
    def predict(self, X):
        """Make predictions for new data"""
        if not self.is_trained:
            raise Exception("Model needs to be trained first!")
        return self.model.predict(X)
    
    def save_model(self, filepath='model.joblib'):
        """Save the trained model"""
        if not self.is_trained:
            raise Exception("Model needs to be trained first!")
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load_model(cls, filepath='model.joblib'):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise Exception(f"Model file {filepath} not found!")
        instance = cls()
        instance.model = joblib.load(filepath)
        instance.is_trained = True
        return instance

if __name__ == "__main__":
    # Test the model with some dummy data
    from preprocessing import load_and_preprocess_data, prepare_data
    
    # Load and prepare data
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Train and evaluate model
    model = SLRModel()
    model.train(X_train, y_train)
    mse, r2 = model.evaluate(X_test, y_test)
    
    # Save the model
    model.save_model('slr_model.joblib') 