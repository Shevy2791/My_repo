#import modules
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn import preprocessing
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly
from scipy.stats import boxcox
from scipy.special import inv_boxcox
pd.set_option('display.max_rows', 1000)
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import ipywidgets as wg 
from Ipython.display import display

#create a connection string for the database

connection_string = 'postgresql://username:password@localhost:5432/your_database'  
# Replace 'username', 'password', and 'your_database' with your actual database credentials
engine = create_engine(connection_string)
#create cursor
connection = engine.connect()

# Define the SQL query to fetch data
query = "SELECT * FROM your_table"


#run query and save to dataframe

with engine.connect() as connection:
    df = pd.read_sql(query, connection, parse dates=True)
df.index.frequency = 'D'  # Set the frequency of the index to daily

filtered_df = df[df['Group Category'] == 'My Group' ]  
#filter the dataframe for a specific group category
#drop duplicate rows if applicable
filtered_df = filtered_df.drop_duplicates()

#drop columns that are not needed, for prophet we only need the date and value columns
filtered_df = filtered_df.drop(columns=['Group Category', 'Group Name', 'Group ID', 'Group Description'])

#Convert the date column to datetime format
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

#Rename the columns to fit the Prophet requirements
filtered_df = filtered_df.rename(columns={'Date': 'ds', 'Value': 'y'})

#view the first few rows of the dataframe
print(filtered_df.head())

#view columns of the dataframe
print(filtered_df.columns)

#check for null values in the dataframe
print(filtered_df.isnull().sum())

# create dataframe for Prophet special dates/holidays
special_dates = pd.DataFrame({
    'holiday': 'special_event',
    'ds': pd.to_datetime(['2023-01-01', '2023-12-25']),  # Example dates
    'lower_window': 0,
    'upper_window': 1
})

# month end dates if applicable
def is_month_end_extended(date):
    day = date.day
    days_in_month = (date + pd.offsets.MonthEnd(0)).day
    if day > days_in_month - 4:
        return 1
    if day <= days_in_month - 4:
        return 1
    return 0
filtered_df['is_month_end_extended'] = filtered_df['ds'].apply(is_month_end_extended)

#create functions for evaluating the model
def getPerformanceMetrics(model):
    return performance_metrics(getCrossValidation(model))

def getCrossValidation(model):
    return cross_validation(model, initial='365 days', period='180 days', horizon='31 days')


#train the Prophet model

model = Prophet(
    holidays = special_dates,
    holidays_prior_scale=0.1) # Adjust the prior scale for holidays
model.add_country_holidays(country_name= 'UK' )  # Add country-specific holidays if applicable
interval_width = 0.95  # Set the prediction interval width
model.add_regressor('is_month_end_extended')  # Add the month end regressor if applicable
model.fit(filtered_df)  # Fit the model to the filtered dataframe

#print model evaluation metrics
print(getPerformanceMetrics(model).mean()
)

#create a future dataframe for predictions
future = model.make_future_dataframe(periods=365, freq='D')  # Extend the future dataframe by 365 days
future['is_month_end_extended'] = future['ds'].apply(is_month_end_extended)  # Add the month end regressor to the future dataframe
# Make predictions using the model
forecast = model.predict(future)
# Plot the forecast using Plotly
fig = plot_plotly(model, forecast)
# Show the plot
fig.show()

#save forecast to excel
forecast.to_excel('forecast.xlsx', index=False)

fig.write_html('forecast.html')  # Save the plot as an HTML file


