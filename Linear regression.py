import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 

df = pd.read_csv("C:/Users/Paulj/Downloads/Real-estate1.csv")
df.drop('No', inplace = True,axis=1) 


sns.scatterplot(x='X4 number of convenience stores', y='Y house price of unit area',data=df)

#X=df.drop('Y house price of unit area', axis=1)
#Y=df['Y house price of unit area']


#X_train, X_test, y_train, y_test = train_test_split( 
    #X, Y, test_size=0.3, random_state=101) 

#mod= LinearRegression()

#mod.fit(X_train,y_train)

#predictions = mod.predict(X_test)

#print('mean_absolute_error:',mean_absolute_error(y_test,predictions))
#print('mean squared error:',mean_squared_error(y_test,predictions))

#r_sq = mod.score(X,Y)

#print(r_sq)

df.rename(columns={
    'Y house price of unit area': 'Y_house_price',
    'X2 house age': 'X2_house_age',
    'X4 number of convenience stores': 'X4_convenience_stores'
}, inplace=True)

s = ols(formula= 'Y_house_price ~ X2_house_age + X4_convenience_stores',data=df).fit()

print(s.summary())


# Plot regression results
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_regress_exog(s, 'X2_house_age', fig=fig)

# Plot regression results for convenience stores as well
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_regress_exog(s, 'X4_convenience_stores', fig=fig)

plt.show()