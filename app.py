from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import joblib
from simple_data_processor import SimpleDataProcessor
from model_trainer import ModelTrainer
from predictor import AccidentPredictor
from dashboard import DashboardGenerator

app = Flask(__name__)

# Global variables
data_processor = None
predictor = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    global data_processor, predictor
    
    if request.method == 'POST':
        try:
            # Initialize data processor
            data_processor = SimpleDataProcessor()
            
            # Check if dataset exists
            rta_path = 'data/RTA Dataset.csv'
            
            if not os.path.exists(rta_path):
                return jsonify({
                    'status': 'error',
                    'message': f'RTA dataset not found at {rta_path}. Please make sure the file exists.'
                })
            
            # Load and preprocess data
            print("Loading and preprocessing RTA dataset...")
            X, y_encoded, y_original = data_processor.load_and_preprocess(rta_path)
            
            # Scale features
            X_scaled = data_processor.scaler.fit_transform(X)
            
            # Train model
            trainer = ModelTrainer()
            best_model, X_test, y_test = trainer.train_models(X_scaled, y_encoded)
            
            # Save model and processor
            os.makedirs('models', exist_ok=True)
            trainer.save_model(best_model, 'models/accident_model.pkl')
            joblib.dump(data_processor, 'models/data_processor.pkl')
            
            # Show feature importance
            trainer.feature_importance(best_model, data_processor.feature_columns)
            
            # Initialize predictor
            predictor = AccidentPredictor('models/accident_model.pkl', data_processor)
            
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully!',
                'accuracy': f"{trainer.best_score:.4f}",
                'dataset_size': len(X)
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Training error: {error_details}")
            return jsonify({
                'status': 'error',
                'message': f'Training failed: {str(e)}'
            })
    
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global predictor
    
    if request.method == 'GET':
        feature_options = {}
        
        # Try to load existing model to get feature options
        try:
            if predictor is None:
                data_processor = joblib.load('models/data_processor.pkl')
                predictor = AccidentPredictor('models/accident_model.pkl', data_processor)
            
            feature_options = predictor.get_feature_options()
        except:
            # If model not trained, provide default options based on RTA dataset
            feature_options = {
                'Day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'Age_band': ['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown'],
                'Gender': ['Male', 'Female', 'Unknown'],
                'Education': ['Junior high school', 'Above high school', 'Elementary school', 'High school', 'Unknown', 'Illiterate'],
                'Experience': ['1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'Below 1yr', 'No Licence', 'Unknown'],
                'Vehicle_Type': ['Automobile', 'Lorry (41?100Q)', 'Public (> 45 seats)', 'Lorry (11?40Q)', 'Public (13?45 seats)', 'Taxi', 'Other', 'Unknown'],
                'Location_Type': ['Residential areas', 'Office areas', 'Other', 'Recreational areas', 'Industrial areas', 'Church areas', 'Market areas', 'Rural village areas', 'Unknown'],
                'Road_Alignment': ['Tangent road with flat terrain', 'Tangent road with mild grade and flat terrain', 'Escarpments', 'Gentle horizontal curve', 'Tangent road with mountainous terrain and', 'Steep grade downward with mountainous terrain', 'Tangent road with rolling terrain', 'Unknown'],
                'Junction_Type': ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown'],
                'Road_Surface': ['Asphalt roads', 'Earth roads', 'Gravel roads', 'Other', 'Asphalt roads with some distress', 'Unknown'],
                'Road_Conditions': ['Dry', 'Wet or damp', 'Snow', 'Other', 'Unknown'],
                'Light_Conditions': ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit', 'Unknown'],
                'Weather_Conditions': ['Normal', 'Raining', 'Cloudy', 'Snow', 'Fog or mist', 'Raining and Windy', 'Other', 'Unknown'],
                'Collision_Type': ['Vehicle with vehicle collision', 'Collision with animals', 'Collision with roadside-parked vehicles', 'Collision with roadside objects', 'Collision with pedestrians', 'Other', 'Rollover', 'Fall from vehicles', 'Unknown'],
                'Accident_Cause': ['No distancing', 'Changing lane to the left', 'Changing lane to the right', 'Moving Backward', 'Overtaking', 'Driving carelessly', 'No priority to vehicle', 'No priority to pedestrian', 'Overloading', 'Other', 'Unknown']
            }
        
        return render_template('predict.html', feature_options=feature_options)
    
    elif request.method == 'POST':
        try:
            if predictor is None:
                # Try to load existing model
                try:
                    data_processor = joblib.load('models/data_processor.pkl')
                    predictor = AccidentPredictor('models/accident_model.pkl', data_processor)
                except:
                    return render_template('results.html', 
                                        prediction='Model Not Trained',
                                        probabilities={},
                                        confidence='0%',
                                        input_features=request.form.to_dict(),
                                        error='Please train the model first.')
            
            # Get form data
            input_features = {}
            feature_fields = ['Day_of_week', 'Age_band', 'Gender', 'Education', 'Experience',
                            'Vehicle_Type', 'Location_Type', 'Road_Alignment', 'Junction_Type',
                            'Road_Surface', 'Road_Conditions', 'Light_Conditions',
                            'Weather_Conditions', 'Collision_Type', 'Accident_Cause']
            
            for field in feature_fields:
                input_features[field] = request.form.get(field, 'Unknown')
            
            # Make prediction
            result = predictor.predict(input_features)
            
            return render_template('results.html', 
                                prediction=result['prediction'],
                                probabilities=result['probabilities'],
                                confidence=result['confidence'],
                                input_features=input_features)
            
        except Exception as e:
            import traceback
            print(f"Prediction error: {traceback.format_exc()}")
            return render_template('results.html', 
                                prediction='Error',
                                probabilities={},
                                confidence='0%',
                                input_features=request.form.to_dict(),
                                error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Model not trained'}), 400
    
    try:
        data = request.get_json()
        result = predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
# Add this import at the top, then replace the dashboard route:

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard with visualizations"""
    try:
        dashboard_gen = DashboardGenerator()
        dashboard_data = dashboard_gen.generate_dashboard_data()
        
        return render_template('dashboard.html', 
                             dashboard_data=dashboard_data,
                             model_loaded=dashboard_data['model_loaded'])
    except Exception as e:
        return f"Dashboard error: {str(e)}"

@app.route('/api/status')
def api_status():
    """API endpoint to check model status"""
    status = {
        'model_trained': predictor is not None,
        'service': 'running'
    }
    
    if predictor is not None:
        status['feature_options'] = list(predictor.get_feature_options().keys())
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def init_app():
    """Initialize the application"""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Try to load existing model
    global predictor
    try:
        if os.path.exists('models/accident_model.pkl') and os.path.exists('models/data_processor.pkl'):
            data_processor = joblib.load('models/data_processor.pkl')
            predictor = AccidentPredictor('models/accident_model.pkl', data_processor)
            print("✓ Pre-trained model loaded successfully")
        else:
            print("ℹ No pre-trained model found. Please train the model first.")
    except Exception as e:
        print(f"⚠ Could not load pre-trained model: {e}")

if __name__ == '__main__':
    # Initialize the application
    init_app()
    
    print("🚀 Starting Traffic Accident Prediction System...")
    print("📍 Application running on: http://127.0.0.1:5000")
    print("📊 Available endpoints:")
    print("   - GET  /              : Home page")
    print("   - GET  /train         : Train model page")
    print("   - POST /train         : Train model API")
    print("   - GET  /predict       : Prediction form")
    print("   - POST /predict       : Make prediction")
    print("   - GET  /dashboard     : Analytics dashboard")
    print("   - GET  /api/status    : API status check")
    print("   - POST /api/predict   : Prediction API")
    
    app.run(debug=True, host='0.0.0.0', port=5000)