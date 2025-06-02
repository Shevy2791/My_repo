import kagglehub

# Download latest version
path = kagglehub.dataset_download("anandshaw2001/imdb-movies-and-tv-shows")

print("Path to dataset files:", path)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)

#import data and assign it to a variable
df = pd.read_csv("C:/Users/Paulj/.cache/kagglehub/datasets/anandshaw2001/imdb-movies-and-tv-shows/versions/2/IMDb 2024 Movies TV Shows.csv")

#view number of columns and rows in data
print(df.shape)

#view top 5 rows

print(df.head())

#show columns
print(df.columns)

#view data types

print(df.dtypes)

# remove columns you don't need

df = df[['Budget', #'Home_Page', 
          'Movie_Name', 'Genres', #'Overview', 
          #'Cast',
       'Original_Language', #'Storyline',
       'Production_Company', #'Release_Date',
       'Revenue', #'Run_Time', #'Tagline',
       'Vote_Average', 'Vote_Count']].copy()

print(df.shape)

print(df.dtypes)

#convert to correct data types

# Function to convert shorthand notations like '151K' to numeric values
def convert_shorthand(value):
    value = str(value).upper().strip()  # Ensure consistent case and remove extra spaces
    if 'K' in value:
        return float(value.replace('K', '')) * 1_000  # Replace 'K' and multiply by 1,000
    elif 'M' in value:
        return float(value.replace('M', '')) * 1_000_000  # Replace 'M' and multiply by 1,000,000
    else:
        return float(value)  # If no shorthand, return as a float

# Apply the function to the 'Vote_Count' column
df['Vote_Count'] = df['Vote_Count'].apply(convert_shorthand)
df['Vote_Count'] = pd.to_numeric(df['Vote_Count'])


# Remove dollar signs and commas, and then convert to numeric
df['Budget'] = df['Budget'].replace({'\$': '', ',': ''}, regex=True)  # Remove $ and ,
df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')  # Convert to numeric, setting invalid parsing as NaN
df['Revenue'] = df['Revenue'].replace({'\$': '', ',': ''}, regex=True)  
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')  


print(df.dtypes)

print(df.head())

#check missing values

print(df.isna().sum())

#check duplicates
print(df.loc[df.duplicated()])

#drop rows with missing values and resetting index

df = df.dropna(subset=['Budget']).reset_index(drop=True).copy()

print(df.isna().sum())

print(df.shape)

#visualising count and basic barchart

ax = df['Vote_Average'].value_counts().head(10).plot(kind='bar', title="Top Ten Average Ratings")

ax.set_xlabel('Average Rating')
ax.set_ylabel('Count')

#plt.show()

#histogram
ax = df['Vote_Average'].plot(kind= 'hist', bins=20, title="Rating Distribution")
ax.set_xlabel('Average Rating')
ax.set_ylabel('Freq')

#plt.show()

#kernal distribution estimate
ax = df['Vote_Average'].plot(kind= 'kde', title="Rating Distribution")
ax.set_xlabel('Average Rating')
ax.set_ylabel('Freq')

#plt.show()


ax = df.plot(kind='scatter',x='Budget',
        y='Revenue',title="Movie Budget vs Revenue")

ax.set_xlabel('Movie Budget')
ax.set_ylabel('Revenue')

#plt.show()

#seaborn scatter function

sns.scatterplot(x='Budget',
        y='Revenue', data=df)

#plt.show()

#sns pairplot

plt.figure(figsize=(12, 8))

sns.pairplot(df,vars=['Budget', 'Revenue', 'Vote_Average'], hue='Genres')

#plt.show()

#correlation

df_corr = df[['Budget','Revenue', 'Vote_Average']].corr()
print(df_corr)

sns.heatmap(df_corr, annot=True)

#plt.show()

df.corr(['Budget'])