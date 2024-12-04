# Expanding the synthetic dataset to include 'NextDayWeatherCondition' as a categorical target
import pandas as pd
import numpy as np

# Set the number of samples
num_samples = 1000

# Generate synthetic weather and air quality data
data = {
    'Temperature': np.random.normal(20, 10, num_samples),       # Mean temp around 20°C
    'Humidity': np.random.uniform(30, 90, num_samples),         # Humidity between 30% and 90%
    'WindSpeed': np.random.uniform(0, 20, num_samples),         # Wind speed in m/s
    'Pressure': np.random.normal(1013, 10, num_samples),        # Pressure in hPa
    'Precipitation': np.random.uniform(0, 50, num_samples),     # Rainfall in mm
    'CloudCover': np.random.uniform(0, 100, num_samples),       # Cloud cover in %
    'DewPoint': np.random.uniform(5, 20, num_samples),          # Dew point in °C
    'SolarRadiation': np.random.uniform(100, 1000, num_samples),# Solar radiation in W/m²
    'Visibility': np.random.uniform(1, 10, num_samples),        # Visibility in km
    'UVIndex': np.random.randint(0, 11, num_samples),           # UV Index 0-10
    'AQI': np.random.randint(0, 300, num_samples),              # Air Quality Index level
    'CO': np.random.uniform(0.1, 2.0, num_samples),             # Carbon Monoxide (ppm)
    'NO2': np.random.uniform(0.1, 0.3, num_samples),            # Nitrogen Dioxide (ppb)
    'SO2': np.random.uniform(0.01, 0.1, num_samples),           # Sulfur Dioxide (ppb)
    'O3': np.random.uniform(0.02, 0.2, num_samples),            # Ozone (ppb)
    'Month': np.random.randint(1, 13, num_samples),             # Month 1-12
    'NextDayTemperature': np.random.normal(20, 10, num_samples) # Target variable (Next Day Temperature)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define weather conditions based on 'NextDayTemperature' and 'Precipitation'
conditions = []
for temp, precip in zip(df['NextDayTemperature'], df['Precipitation']):
    if precip > 30:
        conditions.append("Rainy")
    elif temp > 25:
        conditions.append("Sunny")
    elif 10 <= temp <= 25:
        conditions.append("Cloudy")
    else:
        conditions.append("Cold")

# Add the new 'NextDayWeatherCondition' column
df['NextDayWeatherCondition'] = conditions

# Save the dataset with 'NextDayWeatherCondition'
file_path = "updated_dataset2.csv"
df.to_csv(file_path, index=False)
file_path
