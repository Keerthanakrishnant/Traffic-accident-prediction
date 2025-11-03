import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SimpleDataProcessor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_and_preprocess(self, file_path):
        """Load and preprocess the RTA dataset"""
        print("Loading RTA dataset...")
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Handle missing values
        df = df.fillna('Unknown')
        
        # Select relevant features for prediction
        feature_columns = [
            'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
            'Educational_level', 'Driving_experience', 'Type_of_vehicle',
            'Area_accident_occured', 'Road_allignment', 'Types_of_Junction',
            'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
            'Weather_conditions', 'Type_of_collision', 'Cause_of_accident'
        ]
        
        # Filter only columns that exist in the dataset
        available_columns = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_columns
        print(f"Using features: {available_columns}")
        
        # Prepare features and target
        X = df[available_columns].copy()
        y = df['Accident_severity']
        
        print(f"Target distribution:\n{y.value_counts()}")
        
        # Encode categorical features
        for column in available_columns:
            if X[column].dtype == 'object':
                self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
        
        # Encode target variable
        self.label_encoders['Accident_severity'] = LabelEncoder()
        y_encoded = self.label_encoders['Accident_severity'].fit_transform(y)
        
        return X, y_encoded, y
    
    def preprocess_new_data(self, input_data):
        """Preprocess new data for prediction"""
        processed_data = {}
        
        # Map input names to dataset column names
        column_mapping = {
            'Day_of_week': 'Day_of_week',
            'Age_band': 'Age_band_of_driver',
            'Gender': 'Sex_of_driver',
            'Education': 'Educational_level',
            'Experience': 'Driving_experience',
            'Vehicle_Type': 'Type_of_vehicle',
            'Location_Type': 'Area_accident_occured',
            'Road_Alignment': 'Road_allignment',
            'Junction_Type': 'Types_of_Junction',
            'Road_Surface': 'Road_surface_type',
            'Road_Conditions': 'Road_surface_conditions',
            'Light_Conditions': 'Light_conditions',
            'Weather_Conditions': 'Weather_conditions',
            'Collision_Type': 'Type_of_collision',
            'Accident_Cause': 'Cause_of_accident'
        }
        
        for input_feature, dataset_feature in column_mapping.items():
            if input_feature in input_data and dataset_feature in self.label_encoders:
                value = input_data[input_feature]
                # Handle unseen labels
                if value in self.label_encoders[dataset_feature].classes_:
                    processed_data[dataset_feature] = self.label_encoders[dataset_feature].transform([value])[0]
                else:
                    # Use the first class for unseen labels
                    processed_data[dataset_feature] = 0
            else:
                processed_data[dataset_feature] = 0  # Default value
        
        # Create feature array in correct order
        feature_array = np.array([processed_data.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
        
        return self.scaler.transform(feature_array)
    
    def get_feature_options(self):
        """Get available options for categorical features"""
        options = {}
        for feature, encoder in self.label_encoders.items():
            if feature != 'Accident_severity':
                # Map back to user-friendly names
                user_friendly_name = self.get_user_friendly_name(feature)
                options[user_friendly_name] = list(encoder.classes_)
        return options
    
    def get_user_friendly_name(self, dataset_name):
        """Convert dataset column names to user-friendly names"""
        mapping = {
            'Day_of_week': 'Day_of_week',
            'Age_band_of_driver': 'Age_band',
            'Sex_of_driver': 'Gender',
            'Educational_level': 'Education',
            'Driving_experience': 'Experience',
            'Type_of_vehicle': 'Vehicle_Type',
            'Area_accident_occured': 'Location_Type',
            'Road_allignment': 'Road_Alignment',
            'Types_of_Junction': 'Junction_Type',
            'Road_surface_type': 'Road_Surface',
            'Road_surface_conditions': 'Road_Conditions',
            'Light_conditions': 'Light_Conditions',
            'Weather_conditions': 'Weather_Conditions',
            'Type_of_collision': 'Collision_Type',
            'Cause_of_accident': 'Accident_Cause'
        }
        return mapping.get(dataset_name, dataset_name)