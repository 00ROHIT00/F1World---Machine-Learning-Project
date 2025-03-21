from django.shortcuts import render
import pandas as pd
import numpy as np
import os
import joblib
import json

def mlr_view(request):
    context = {
        'feature_names': ['Qualifying Position', 'Number of Overtakes', 'Pit Stops', 'Fastest Lap Bonus'],
        'target_name': 'Points Scored'
    }

    # Load model metrics for R² score
    try:
        with open('main/MLR/models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
            context['r2_score'] = metrics['r2_score']
    except Exception as e:
        context['error'] = f"Error loading model metrics: {str(e)}"
        return render(request, 'main/mlr.html', context)

    if request.method == 'POST':
        try:
            # Get input values
            input_values = {}
            for i, feature in enumerate(context['feature_names']):
                input_values[feature] = float(request.POST.get(f'feature_{i}'))

            # Load model and scaler
            model = joblib.load('main/MLR/models/mlr_model.joblib')
            scaler = joblib.load('main/MLR/models/scaler.joblib')

            # Prepare input for prediction
            X = pd.DataFrame([input_values])
            X_scaled = scaler.transform(X)

            # Make prediction and format it properly
            prediction = model.predict(X_scaled)[0]
            prediction = max(0, prediction)  # Ensure prediction is not negative
            prediction = round(prediction, 1)  # Round to 1 decimal place
            # Convert to string and remove .0 if present
            prediction_str = str(prediction).rstrip('0').rstrip('.')

            # Update context with results
            context.update({
                'success': True,
                'prediction': prediction_str,
                'input_values': input_values
            })

        except Exception as e:
            context['error'] = f"Prediction error: {str(e)}"

    return render(request, 'main/mlr.html', context) 