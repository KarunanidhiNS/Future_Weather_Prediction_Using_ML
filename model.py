import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('dataset/dataset.csv')

features = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure', 'Precipitation', 
            'CloudCover', 'DewPoint', 'SolarRadiation', 'Visibility', 'UVIndex', 
            'AQI', 'CO', 'NO2', 'SO2', 'O3']
target_temperature = 'NextDayTemperature'
target_weather = 'NextDayWeatherCondition'

X = data[features]
y_temperature = data[target_temperature]
y_weather = data[target_weather]

X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y_temperature, test_size=0.2, random_state=42)
_, _, y_train_weather, y_test_weather = train_test_split(X, y_weather, test_size=0.2, random_state=42)

model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
model_weather = RandomForestClassifier(n_estimators=100, random_state=42)

model_temp.fit(X_train, y_train_temp)

y_pred_temp = model_temp.predict(X_test)

mae = mean_absolute_error(y_test_temp, y_pred_temp)
print(f'Mean Absolute Error for Next Day Temperature Prediction: {mae}')

model_weather.fit(X_train, y_train_weather)

y_pred_weather = model_weather.predict(X_test)

accuracy = accuracy_score(y_test_weather, y_pred_weather)
print(f'Accuracy for Next Day Weather Condition Prediction: {accuracy * 100:.2f}%')

joblib.dump(model_temp, 'rf_temp_model.pkl')
joblib.dump(model_weather, 'rf_weather_model.pkl')

scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler.pkl')
