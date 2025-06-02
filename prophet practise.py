import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error 

#import csv as dataframe
df = pd.read_csv("C:\\Users\\Paulj\\Downloads\\archive (3)\\DailyDelhiClimateTrain.csv")

#view contents
print(df.head())

#rename columns to match Prophet's requirements
#'date' to 'ds' and 'meantemp' to 'y'
df.rename(columns={'date': 'ds', 'meantemp': 'y'}, inplace=True)
df.index = pd.to_datetime(df['ds'])
print(df.head())

#plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'])
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()

#split data into train and test sets
split_date = '2015-12-31'
train = df[df['ds'] <= split_date].copy()   
test = df[df['ds'] > split_date].copy()
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

#plot the train-test split
plt.figure(figsize=(10, 6))
plt.plot(train['ds'], train['y'], label='Training Data')
plt.plot(test['ds'], test['y'], label='Testing Data', color='orange')
plt.title('Train-Test Split')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

#fit model to training data
model = Prophet(interval_width=0.95, yearly_seasonality=True, changepoint_prior_scale=0.05)
model.add_country_holidays(country_name='India')
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.add_seasonality(name='weekly', period=7, fourier_order=3)
model.fit(train)

#forecast future values
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

#merge test and forecast data
test = test.reset_index(drop=True)

test['ds'] = pd.to_datetime(test['ds'])
forecast['ds'] = pd.to_datetime(forecast['ds'])

results = test.merge(forecast[['ds', 'yhat']], on='ds', how='left')

#evaluate model performance
mae = mean_absolute_error(results['y'], results['yhat'])
mse = mean_squared_error(results['y'], results['yhat'])
print(f"MAE: {mae}, MSE: {mse}")

#plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(results['ds'], results['y'], label='Actual', color='blue')
plt.plot(results['ds'], results['yhat'], label='Predicted', color='red')
plt.title('Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()