from django.shortcuts import render
import pandas as pd
import numpy as np
import os
import joblib
import json

def slr_view(request):
    context = {
        'feature_names': ['Number of Tires Changed'],
        'target_name': 'Pit Stop Time (s)'
    }

    # Load model metrics for R² score
    try:
        with open('main/SLR/models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
            context['r2_score'] = metrics['r2_score']
    except Exception as e:
        context['error'] = f"Error loading model metrics: {str(e)}"
        return render(request, 'main/slr.html', context)

    if request.method == 'POST':
        try:
            # Get input values
            input_values = {}
            for i, feature in enumerate(context['feature_names']):
                value = int(request.POST.get(f'feature_{i}'))
                if value < 1 or value > 4:
                    raise ValueError("Number of tires must be between 1 and 4")
                input_values[feature] = value

            # Load model and scaler
            model = joblib.load('main/SLR/models/slr_model.joblib')
            scaler = joblib.load('main/SLR/models/scaler.joblib')

            # Create DataFrame with the exact same structure as training
            X = pd.DataFrame(columns=['Number of Tires Changed'])
            X.loc[0, 'Number of Tires Changed'] = input_values['Number of Tires Changed']
            X_scaled = scaler.transform(X)

            # Make prediction and format it properly
            prediction = model.predict(X_scaled)[0]
            prediction = max(1.9, prediction)  # Ensure minimum pit stop time is 1.9 seconds
            prediction = round(prediction, 2)  # Round to 2 decimal places for seconds

            # Update context with results
            context.update({
                'success': True,
                'prediction': f"{prediction:.2f}",  # Always show 2 decimal places for seconds
                'input_values': input_values
            })

        except ValueError as ve:
            context['error'] = str(ve)
        except Exception as e:
            context['error'] = f"Prediction error: {str(e)}"

    return render(request, 'main/slr.html', context) 