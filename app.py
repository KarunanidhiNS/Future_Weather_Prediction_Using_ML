from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model_temp = joblib.load('models/rf_temp_model.pkl')
model_weather = joblib.load('models/rf_weather_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            'temperature', 'humidity', 'windspeed', 'pressure', 'precipitation', 
            'cloudcover', 'dewpoint', 'solarradiation', 'visibility', 'uvindex', 
            'aqi', 'co', 'no2', 'so2', 'o3'
        ]

        input_data = [
            float(request.form[feature]) for feature in features
        ]
        
        scaled_data = scaler.transform([input_data])

        next_day_temp = model_temp.predict(scaled_data)[0]
        next_day_weather = model_weather.predict(scaled_data)[0]

        return render_template('result.html', 
                               next_day_temperature=next_day_temp,
                               next_day_weather=next_day_weather)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
