#  Traffic Accident Severity Prediction System

A comprehensive machine learning system that predicts traffic accident severity using Gradient Boosting algorithms and provides real-time predictions through a Flask web interface.

##  Project Overview

This system analyzes historical traffic accident data to predict accident severity levels and identify high-risk scenarios. The model achieves **87.3% accuracy** with strong performance in predicting severe accidents (**82.1% precision**, **79.8% recall**).


##  Features

- **Real-time Accident Severity Prediction**
- **Interactive Data Dashboard**
- **Advanced Feature Engineering**
- **Multiple Machine Learning Models**
- **Web-based User Interface**
- **Geographic Risk Analysis**

##  Project Structure
traffic_accident_prediction/
-├── app.py # Flask web application
├── model_trainer.py # Machine learning model training
├── data_processor.py # Data preprocessing pipeline
├── predictor.py # Real-time prediction engine
├── dashboard.py # Interactive visualization
├── requirements.txt # Python dependencies
├── models/
│ ├── accident_model.pkl # Trained ML model
│ └── data_processor.pkl # Preprocessing pipeline
├── templates/ # HTML templates
│ ├── index.html # Home page
│ ├── predict.html # Prediction interface
│ ├── results.html # Results display
│ └── dashboard.html # Analytics dashboard
├── static/
│ └── style.css # Custom styling
└── data/ # Dataset directory

##  Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone Repository
```bash
git clone https://github.com/keerthanakrishnant/traffic-accident-prediction.git
cd traffic-accident-prediction
pip install -r requirements.txt
python model_trainer.py
python app.py
