#!/usr/bin/python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def sns_jointplot(x,y,frame,sort):

    if sort == "s":
        kind = "scatter"

    else:
        kind = "hex"

    #plt.figure()
    sns.jointplot(frame[x], frame[y], kind=kind)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(str("Ecommerce Data"))

data = pd.read_csv("Ecommerce Customers")

for i in (data, data.head(), data.describe(), data.info()):
    print(i)

# sns_jointplot("Time on Website", "Yearly Amount Spent", data,"s")
# sns_jointplot("Time on App", "Yearly Amount Spent", data,"s")
# sns_jointplot("Time on App", "Length of Membership", data, "hex")

#sns.pairplot(data)
#sns.lmplot("Yearly Amount Spent", "Length of Membership", data)

x = data[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = data["Yearly Amount Spent"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(x_train, y_train)

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficients'])

predictions = lm.predict(x_test)
#plt.scatter(y_test, predictions)

print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot((y_test - predictions), bins=50)
plt.show()