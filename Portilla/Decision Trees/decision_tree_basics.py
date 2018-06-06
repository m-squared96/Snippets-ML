#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def initial_explore(df):
    print(data.info)
    print(data.head)

    pp_option = str(input("Would you like to produce a Seaborn Pairplot of the data?(y/n)    "))

    if pp_option.lower() == "y" or pp_option.lower() == "yes":
        sns.pairplot(df, hue="Kyphosis")

def preprocess(df):
    x = df.drop('Kyphosis', axis=1)
    y = df['Kyphosis']

    return train_test_split(x,y,test_size=0.3)

def analysis_dtree(xtr, xte, ytr, yte):
    dtree = DecisionTreeClassifier()
    dtree.fit(xtr, ytr)
    predictions = dtree.predict(xte)

    print("Decision Tree Model:")
    print("\n", confusion_matrix(yte, predictions))
    print("\n", classification_report(yte, predictions))

    return dtree

def analysis_rfc(xtr, xte, ytr, yte):
    rfc = RandomForestClassifier()
    rfc.fit(xtr, ytr)
    predictions = rfc.predict(xte)

    print("Random Forest Model:")
    print("\n", confusion_matrix(yte, predictions))
    print("\n", classification_report(yte, predictions))

    return rfc

data = pd.read_csv("kyphosis.csv")

initial_explore(data)
xtr, xte, ytr, yte = preprocess(data)
dtree = analysis_dtree(xtr, xte, ytr, yte)
rfc = analysis_rfc(xtr, xte, ytr, yte)

plt.show()
