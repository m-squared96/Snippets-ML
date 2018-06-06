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
    print(data.columns)

    pp_option = False
    pp_option = str(input("Would you like to produce a histogram of the data?(y/n)    "))

    if pp_option.lower() == "y" or pp_option.lower() == "yes":
        plt.figure()
        df[df['credit.policy'] == 1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')
        df[df['credit.policy'] == 0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit.Policy=0')
        plt.legend()
        plt.xlabel('FICO')

        plt.figure()
        df[df['not.fully.paid'] == 0]['fico'].hist(alpha=0.5,color='green',bins=30,label='Fully Paid')
        df[df['not.fully.paid'] == 1]['fico'].hist(alpha=0.5,color='orange',bins=30,label='Not Fully Paid')
        plt.legend()
        plt.xlabel('FICO')

        plt.figure()
        sns.countplot(df['purpose'], hue=df['not.fully.paid'])
        plt.xlabel('Purpose')
        plt.ylabel('Count')

        plt.figure()
        sns.jointplot(df['fico'], df['int.rate'])

        plt.figure(figsize=(11,7))
        sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',col='not.fully.paid',palette='Set1')

def preprocessing(df):
    cat_feats = ['purpose']
    final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)
    print(final_data.info)

    x = final_data.drop('not.fully.paid', axis=1)
    y = final_data['not.fully.paid']

    return train_test_split(x,y,test_size=0.3, random_state=101)

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

data = pd.read_csv("loan_data.csv")

initial_explore(data)
xtr, xte, ytr, yte = preprocessing(data)
dtree = analysis_dtree(xtr, xte, ytr, yte)
rfc = analysis_rfc(xtr, xte, ytr, yte)

plt.show()
