#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def initial_explore(df):
    print(data.head)
    sns.pairplot(df, hue="TARGET CLASS")

def preprocess(df):
    scaler = StandardScaler()
    scaler.fit(df.drop('TARGET CLASS', axis=1))

    scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

    data_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

    X_train, X_test, y_train, y_test = train_test_split(data_feat, df['TARGET CLASS'], test_size=0.3, random_state=101)

    return X_train, X_test, y_train, y_test

def analyse(xtr, xte, ytr, yte):
    error_rate = []
    max = int(input("Enter max k value to be tested:    "))
    max += 1

    for i in range(1,max):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xtr,ytr)

        pred_i = knn.predict(xte)
        error_rate.append(np.mean(pred_i != yte))

    plt.figure()
    plt.plot(range(1,max), error_rate, ls='--', marker='o', markerfacecolor='r')

    knn = KNeighborsClassifier(n_neighbors=(np.argmin(error_rate)+1))
    knn.fit(xtr,ytr)
    pred = knn.predict(xte)

    print("Best K Value: " + str(np.argmin(error_rate)+1))
    print("\n",confusion_matrix(yte, pred))
    print("\n",classification_report(yte, pred))

data = pd.read_csv("KNN_Project_Data")
initial_explore(data)
X_train, X_test, y_train, y_test = preprocess(data)
analyse(X_train, X_test, y_train, y_test)

plt.show()
