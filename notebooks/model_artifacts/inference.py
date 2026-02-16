import os
import json
import joblib
import numpy as np

def model_fn(model_dir):
    """Load model with sklearn 1.4.x support"""
    import sklearn
    print(f"Loading model with sklearn {sklearn.__version__}")
    
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    return {'model': model, 'scaler': scaler, 'metadata': metadata}

def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        # Handle both string and bytes
        if isinstance(request_body, bytes):
            request_body = request_body.decode('utf-8')
        
        data = json.loads(request_body)
        
        # Handle double-encoded JSON string
        if isinstance(data, str):
            data = json.loads(data)
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        # Convert list of dicts to numpy array
        return np.array([list(d.values()) for d in data])
    raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(input_data, model_dict):
    X_scaled = model_dict['scaler'].transform(input_data)
    predictions = model_dict['model'].predict(X_scaled)
    scores = model_dict['model'].decision_function(X_scaled)
    
    results = []
    for pred, score in zip(predictions, scores):
        results.append({
            'is_anomaly': int(pred == -1),
            'anomaly_score': float(score),
            'confidence': float(abs(score))
        })
    return results

def output_fn(prediction, response_content_type='application/json'):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    raise ValueError(f'Unsupported content type: {response_content_type}')
