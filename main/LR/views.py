from django.shortcuts import render
import pandas as pd
import numpy as np
import os
import joblib
import json

def lr_view(request):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')

    context = {
        'feature_names': ['Qualifying Position', 'Past Race Performance', 'Track Characteristics'],
        'target_name': 'Podium Finish Probability'
    }

    # Load model metrics
    try:
        metrics_path = os.path.join(models_dir, 'model_metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            context.update({
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })
    except Exception as e:
        context['error'] = f"Error loading model metrics: {str(e)}"
        return render(request, 'main/lr.html', context)

    if request.method == 'POST':
        try:
            # Get input values
            qualifying_position = int(request.POST.get('qualifying_position'))
            past_performance = float(request.POST.get('past_performance'))
            track_characteristics = float(request.POST.get('track_characteristics'))

            # Validate inputs
            if qualifying_position < 1 or qualifying_position > 20:
                raise ValueError("Qualifying position must be between 1 and 20")
            if past_performance < 0 or past_performance > 1:
                raise ValueError("Past race performance must be between 0 and 1")
            if track_characteristics < 0 or track_characteristics > 1:
                raise ValueError("Track characteristics must be between 0 and 1")

            # Load model and scaler
            model_path = os.path.join(models_dir, 'lr_model.joblib')
            scaler_path = os.path.join(models_dir, 'scaler.joblib')
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Create DataFrame with the exact same structure as training
            X = pd.DataFrame([[qualifying_position, past_performance, track_characteristics]], 
                           columns=['Qualifying Position', 'Past Race Performance', 'Track Characteristics'])
            
            # Scale features
            X_scaled = scaler.transform(X)

            # Make prediction
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            podium_probability = probabilities[1]  # Probability of getting a podium

            # Update context with results
            context.update({
                'success': True,
                'prediction': 'Podium' if prediction == 1 else 'No Podium',
                'probability': f"{podium_probability:.1%}",
                'input_values': {
                    'qualifying_position': qualifying_position,
                    'past_performance': past_performance,
                    'track_characteristics': track_characteristics
                }
            })

        except ValueError as ve:
            context['error'] = str(ve)
        except Exception as e:
            context['error'] = f"Prediction error: {str(e)}"

    return render(request, 'main/lr.html', context) 