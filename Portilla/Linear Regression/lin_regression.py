#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def plotter(df, x_column, y_column):
    
    plt.figure()
    plt.scatter(df[x_column], df[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title("USA Housing Data")
    
def distplotter(df, x_column, y_column):

    plt.figure()
    sns.distplot(df[x_column])

    plt.figure()
    sns.distplot(df[y_column])

frame = pd.read_csv("USA_Housing.csv")

x = frame[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = frame['Price']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(x_train, y_train)

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficients'])

predictions = lm.predict(x_test)
plt.scatter(y_test, predictions)

print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.show()