import kagglehub

# Download latest version
path = kagglehub.dataset_download("hopesb/student-depression-dataset")

print("Path to dataset files:", path)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)
import sklearn as sk


#import data and assign it to a variable
df = pd.read_csv("C:/Users/Paulj/.cache/kagglehub/datasets/asinow/car-price-dataset/versions/1/car_price_dataset.csv")

#view number of columns and rows in data
print(df.shape)

print(df.head())

print(df.columns)

print(df.dtypes)

df = df[[#'Brand', 'Model', 
         'Year', 'Engine_Size', #'Fuel_Type', 'Transmission',
       'Mileage', 'Doors', 'Owner_Count', 'Price']]

print(df.shape)

price_corr =df.corr()['Price']

print(price_corr)

#scatter plot
plt.scatter(df['Year'], df['Price'])
plt.xlabel("Year")
plt.ylabel("Price")
plt.show()


plt.scatter(df['Mileage'], df['Price'])
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.show()

plt.scatter(df['Engine_Size'], df['Price'])
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.show()

df_np = df.to_numpy()
print(df_np.shape)

x_train, y_train = df_np[:,:3],df_np[:,-1]
print(x_train.shape,y_train.shape)

print(x_train)

from sklearn.linear_model import LinearRegression

sklearn_model = LinearRegression().fit(x_train,y_train)
sklearn_y_predictions = sklearn_model.predict(x_train)

print(sklearn_y_predictions.shape)

from sklearn.metrics import mean_absolute_error, mean_squared_error

print(mean_absolute_error(sklearn_y_predictions, y_train), mean_squared_error(sklearn_y_predictions, y_train))