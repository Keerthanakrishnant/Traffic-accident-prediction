import pandas as pd
import numpy as np
import joblib

class AccidentPredictor:
    def __init__(self, model_path, data_processor):
        self.model = joblib.load(model_path)
        self.data_processor = data_processor
        
    def predict(self, input_features):
        """Predict accident severity for new data"""
        try:
            # Preprocess the input features
            processed_features = self.data_processor.preprocess_new_data(input_features)
            
            # Make prediction
            prediction_encoded = self.model.predict(processed_features)[0]
            prediction_proba = self.model.predict_proba(processed_features)[0]
            
            # Decode prediction
            severity_prediction = self.data_processor.label_encoders['Accident_severity'].inverse_transform([prediction_encoded])[0]
            
            # Get probabilities for all classes
            severity_classes = self.data_processor.label_encoders['Accident_severity'].classes_
            probabilities = {cls: f"{prob*100:.2f}%" for cls, prob in zip(severity_classes, prediction_proba)}
            
            return {
                'prediction': severity_prediction,
                'probabilities': probabilities,
                'confidence': f"{max(prediction_proba)*100:.2f}%"
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'prediction': 'Unknown',
                'probabilities': {},
                'confidence': '0%',
                'error': str(e)
            }
    
    def get_feature_options(self):
        """Get available options for categorical features"""
        return self.data_processor.get_feature_options()