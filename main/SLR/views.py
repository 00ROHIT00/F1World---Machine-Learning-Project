from django.shortcuts import render
import pandas as pd
import numpy as np
import os
import joblib

def slr_view(request):
    context = {}
    
    try:
        # Get the absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(current_dir), 'data', 'SLR.xlsx')
        model_path = os.path.join(current_dir, 'models', 'slr_model.joblib')
        scaler_path = os.path.join(current_dir, 'models', 'scaler.joblib')
        
        # Load the data
        df = pd.read_excel(data_path)
        feature_names = df.columns[:-1].tolist()
        target_name = df.columns[-1]
        
        if request.method == 'POST':
            try:
                # Get input values
                input_values = [float(request.POST.get(f'feature_{i}', 0)) 
                              for i in range(len(feature_names))]
                
                # Load the model and scaler
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Prepare input for prediction
                X_input = np.array(input_values).reshape(1, -1)
                X_input_scaled = scaler.transform(X_input)
                
                # Make prediction
                prediction = model.predict(X_input_scaled)[0]
                
                context['prediction'] = f"{prediction:.2f}"
                context['input_values'] = dict(zip(feature_names, input_values))
                context['success'] = True
                
            except Exception as e:
                context['error'] = f"Prediction error: {str(e)}"
        
        context['feature_names'] = feature_names
        context['target_name'] = target_name
        
    except Exception as e:
        context['error'] = f"Failed to load data: {str(e)}"
    
    return render(request, 'main/slr.html', context) 