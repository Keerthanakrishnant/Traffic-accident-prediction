import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import render_template
import os
import joblib
import numpy as np
from io import BytesIO
import base64

class DashboardGenerator:
    def __init__(self):
        self.data_processor = None
        self.model = None
        
    def load_data_and_model(self):
        """Load the trained model and data processor"""
        try:
            self.data_processor = joblib.load('models/data_processor.pkl')
            self.model = joblib.load('models/accident_model.pkl')
            return True
        except:
            return False
    
    def create_severity_distribution_chart(self):
        """Create severity distribution chart"""
        plt.figure(figsize=(10, 6))
        
        # Sample data - in real app, this would come from the dataset
        severity_counts = {
            'Slight Injury': 65,
            'Serious Injury': 25, 
            'Fatal Injury': 10
        }
        
        colors = ['#4CAF50', '#FF9800', '#F44336']
        plt.bar(severity_counts.keys(), severity_counts.values(), color=colors)
        plt.title('Accident Severity Distribution')
        plt.ylabel('Number of Accidents')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return self.plot_to_base64()
    
    def create_time_distribution_chart(self):
        """Create time distribution chart"""
        plt.figure(figsize=(12, 6))
        
        # Sample time distribution
        time_slots = ['00-03', '03-06', '06-09', '09-12', '12-15', '15-18', '18-21', '21-24']
        accidents_by_time = [5, 3, 15, 20, 18, 25, 22, 10]
        
        plt.plot(time_slots, accidents_by_time, marker='o', linewidth=2, markersize=8)
        plt.fill_between(time_slots, accidents_by_time, alpha=0.3)
        plt.title('Accidents by Time of Day')
        plt.xlabel('Time of Day')
        plt.ylabel('Number of Accidents')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return self.plot_to_base64()
    
    def create_weather_impact_chart(self):
        """Create weather impact chart"""
        plt.figure(figsize=(10, 6))
        
        weather_data = {
            'Normal': 40,
            'Raining': 25,
            'Cloudy': 15,
            'Snow': 10,
            'Fog': 10
        }
        
        colors = ['#4CAF50', '#2196F3', '#FFC107', '#9E9E9E', '#607D8B']
        plt.pie(weather_data.values(), labels=weather_data.keys(), autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Accidents by Weather Conditions')
        plt.tight_layout()
        
        return self.plot_to_base64()
    
    def create_cause_analysis_chart(self):
        """Create cause analysis chart"""
        plt.figure(figsize=(12, 8))
        
        causes = {
            'No Distancing': 35,
            'Changing Lane': 28,
            'Overtaking': 22,
            'Careless Driving': 18,
            'Overspeeding': 15,
            'Other': 30
        }
        
        # Sort by values
        sorted_causes = dict(sorted(causes.items(), key=lambda x: x[1]))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_causes)))
        plt.barh(list(sorted_causes.keys()), list(sorted_causes.values()), color=colors)
        plt.title('Top Accident Causes')
        plt.xlabel('Number of Accidents')
        plt.tight_layout()
        
        return self.plot_to_base64()
    
    def create_feature_importance_chart(self):
        """Create feature importance chart"""
        plt.figure(figsize=(12, 8))
        
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            feature_names = self.data_processor.feature_columns
            
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1]
            
            plt.bar(range(len(feature_importance)), feature_importance[indices])
            plt.xticks(range(len(feature_importance)), 
                      [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.title('Feature Importance in Accident Prediction')
            plt.ylabel('Importance')
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        return self.plot_to_base64()
    
    def plot_to_base64(self):
        """Convert matplotlib plot to base64 string"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graphic = base64.b64encode(image_png).decode('utf-8')
        plt.close()
        
        return graphic
    
    def generate_dashboard_data(self):
        """Generate all dashboard data and charts"""
        model_loaded = self.load_data_and_model()
        
        dashboard_data = {
            'model_loaded': model_loaded,
            'charts': {}
        }
        
        if model_loaded:
            dashboard_data['charts']['severity'] = self.create_severity_distribution_chart()
            dashboard_data['charts']['time'] = self.create_time_distribution_chart()
            dashboard_data['charts']['weather'] = self.create_weather_impact_chart()
            dashboard_data['charts']['causes'] = self.create_cause_analysis_chart()
            dashboard_data['charts']['features'] = self.create_feature_importance_chart()
            
            # Add some statistics
            dashboard_data['stats'] = {
                'total_accidents': 1000,
                'slight_injuries': 650,
                'serious_injuries': 250,
                'fatal_injuries': 100,
                'avg_severity': 'Slight Injury',
                'most_common_cause': 'No Distancing',
                'peak_time': '15-18'
            }
        else:
            # Generate sample charts even if model not loaded
            dashboard_data['charts']['severity'] = self.create_severity_distribution_chart()
            dashboard_data['charts']['time'] = self.create_time_distribution_chart()
            dashboard_data['charts']['weather'] = self.create_weather_impact_chart()
            dashboard_data['charts']['causes'] = self.create_cause_analysis_chart()
            
            dashboard_data['stats'] = {
                'total_accidents': 'N/A (Train model first)',
                'slight_injuries': 'N/A',
                'serious_injuries': 'N/A',
                'fatal_injuries': 'N/A',
                'avg_severity': 'N/A',
                'most_common_cause': 'N/A',
                'peak_time': 'N/A'
            }
        
        return dashboard_data