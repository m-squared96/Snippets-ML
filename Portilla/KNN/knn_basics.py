#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def preprocess(data):
    scaler = StandardScaler()
    scaler.fit(data.drop('TARGET CLASS', axis=1))

    scaled_features = scaler.transform(data.drop('TARGET CLASS', axis=1))

    data_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])

    X_train, X_test, y_train, y_test = train_test_split(data_feat, data['TARGET CLASS'], test_size=0.3, random_state=101)

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
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=(np.argmin(error_rate)+1))
    knn.fit(xtr,ytr)
    pred = knn.predict(xte)

    print("Best K Value: " + str(np.argmin(error_rate)+1))
    print("\n",confusion_matrix(yte, pred))
    print("\n",classification_report(yte, pred))

df = pd.read_csv("Classified Data")
X_train, X_test, y_train, y_test = preprocess(df)
analyse(X_train, X_test, y_train, y_test)
