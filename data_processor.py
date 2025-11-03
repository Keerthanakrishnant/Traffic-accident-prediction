import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.combined_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_datasets(self, rta_path, us_accidents_path):
        """Load and combine both datasets"""
        print("Loading datasets...")
        
        # Load RTA dataset
        rta_df = pd.read_csv(rta_path)
        print(f"RTA Dataset shape: {rta_df.shape}")
        
        # Load US Accidents dataset (sample for demonstration)
        # In practice, you might want to sample this large dataset
        us_df = pd.read_csv(us_accidents_path, nrows=50000)  # Adjust as needed
        print(f"US Accidents Dataset shape: {us_df.shape}")
        
        return rta_df, us_df
    
    def preprocess_rta_data(self, df):
        """Preprocess RTA dataset"""
        rta_data = df.copy()
        
        # Handle missing values
        rta_data = rta_data.fillna('Unknown')
        
        # Select and rename columns to match common schema
        rta_processed = pd.DataFrame()
        
        # Map RTA columns to common features
        column_mapping = {
            'Time': 'Time',
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
            'Cause_of_accident': 'Accident_Cause',
            'Accident_severity': 'Severity'
        }
        
        for new_col, old_col in column_mapping.items():
            if old_col in rta_data.columns:
                rta_processed[new_col] = rta_data[old_col]
            else:
                rta_processed[new_col] = 'Unknown'
        
        rta_processed['Dataset_Source'] = 'RTA'
        rta_processed['Country'] = 'International'
        
        return rta_processed
    
    def preprocess_us_data(self, df):
        """Preprocess US Accidents dataset"""
        us_data = df.copy()
        
        # Handle missing values
        us_data = us_data.fillna('Unknown')
        
        # Select and map US Accidents columns to common schema
        us_processed = pd.DataFrame()
        
        # Map US Accidents columns (adjust based on actual column names)
        # These are example mappings - adjust based on actual US Accidents dataset structure
        column_mapping = {
            'Start_Time': 'Time',
            'Severity': 'Severity',  # This might need transformation
            'Weather_Condition': 'Weather_Conditions',
            'Sunrise_Sunset': 'Light_Conditions',
            'Civil_Twilight': 'Light_Conditions',  # Alternative
        }
        
        # Add more mappings based on actual US Accidents dataset structure
        if 'Severity' in us_data.columns:
            # Convert numeric severity to categorical
            severity_map = {1: 'Slight Injury', 2: 'Slight Injury', 
                          3: 'Serious Injury', 4: 'Fatal Injury'}
            us_processed['Severity'] = us_data['Severity'].map(severity_map).fillna('Slight Injury')
        else:
            us_processed['Severity'] = 'Slight Injury'
            
        # Add other common features with placeholder values
        common_features = ['Time', 'Day_of_week', 'Age_band', 'Gender', 'Education', 
                         'Experience', 'Vehicle_Type', 'Location_Type', 'Road_Alignment',
                         'Junction_Type', 'Road_Surface', 'Road_Conditions', 
                         'Light_Conditions', 'Weather_Conditions', 'Collision_Type', 
                         'Accident_Cause']
        
        for feature in common_features:
            if feature in us_data.columns:
                us_processed[feature] = us_data[feature]
            else:
                us_processed[feature] = 'Unknown'
        
        us_processed['Dataset_Source'] = 'US_Accidents'
        us_processed['Country'] = 'USA'
        
        return us_processed
    
    def combine_datasets(self, rta_path, us_accidents_path):
        """Combine both datasets into a unified format"""
        rta_df, us_df = self.load_datasets(rta_path, us_accidents_path)
        
        # Preprocess both datasets
        rta_processed = self.preprocess_rta_data(rta_df)
        us_processed = self.preprocess_us_data(us_df)
        
        # Combine datasets
        common_columns = list(set(rta_processed.columns) & set(us_processed.columns))
        self.combined_data = pd.concat([
            rta_processed[common_columns],
            us_processed[common_columns]
        ], ignore_index=True)
        
        print(f"Combined dataset shape: {self.combined_data.shape}")
        print(f"Dataset distribution:")
        print(self.combined_data['Dataset_Source'].value_counts())
        print(f"Severity distribution:")
        print(self.combined_data['Severity'].value_counts())
        
        return self.combined_data
    
    def prepare_features(self, data):
        """Prepare features for model training"""
        print("Preparing features...")
        
        # Select features for model training
        feature_columns = [
            'Day_of_week', 'Age_band', 'Gender', 'Education', 'Experience',
            'Vehicle_Type', 'Location_Type', 'Road_Alignment', 'Junction_Type',
            'Road_Surface', 'Road_Conditions', 'Light_Conditions', 
            'Weather_Conditions', 'Collision_Type', 'Accident_Cause', 'Country'
        ]
        
        # Use only columns that exist in the data
        available_columns = [col for col in feature_columns if col in data.columns]
        self.feature_columns = available_columns
        
        X = data[available_columns].copy()
        y = data['Severity']
        
        # Handle missing values
        X = X.fillna('Unknown')
        
        # Encode categorical features
        for column in available_columns:
            if X[column].dtype == 'object':
                self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
        
        # Encode target variable
        self.label_encoders['Severity'] = LabelEncoder()
        y_encoded = self.label_encoders['Severity'].fit_transform(y)
        
        return X, y_encoded, y
    
    def preprocess_new_data(self, input_data):
        """Preprocess new data for prediction"""
        processed_data = {}
        
        for feature in self.feature_columns:
            if feature in input_data:
                value = input_data[feature]
                if feature in self.label_encoders:
                    # Handle unseen labels
                    if value in self.label_encoders[feature].classes_:
                        processed_data[feature] = self.label_encoders[feature].transform([value])[0]
                    else:
                        # Use the most frequent class for unseen labels
                        processed_data[feature] = 0
                else:
                    processed_data[feature] = value
            else:
                processed_data[feature] = 0  # Default value
        
        # Create feature array in correct order
        feature_array = np.array([processed_data[col] for col in self.feature_columns]).reshape(1, -1)
        
        return self.scaler.transform(feature_array)