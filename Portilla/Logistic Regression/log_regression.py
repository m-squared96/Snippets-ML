#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style('whitegrid')

def initial_analysis(data, title_text):

    for i in (data.head(), data.describe(), data.info()):
        print(i)

    plt.figure()
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False)
    plt.title(title_text)
    
    try:
        plt.figure()
        sns.countplot(x='Survived', hue="Sex",  data=data, palette="RdBu_r")
        plt.title(title_text)

        plt.figure()
        sns.countplot(x='Survived', hue="Pclass",  data=data)
        plt.title(title_text)

    except:
        pass

    plt.figure()
    sns.distplot(data['Age'].dropna(), kde=False, bins=30)
    plt.title(title_text)

    plt.figure()
    sns.countplot(x='SibSp', data=data)
    plt.title(title_text)

    plt.figure()
    sns.distplot(data["Fare"], kde=False, bins=30)
    plt.xlim(0,max(data["Fare"]))
    plt.title(title_text)

    plt.figure(figsize=(10,7))
    sns.boxplot(x="Pclass", y="Age", data=data)
    plt.title(title_text)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29
        
        elif Pclass == 3:
            return 24

    else:
        return Age

def cleaner(data):
    data["Age"] = data[["Age", "Pclass"]].apply(impute_age, axis=1)
    data.drop("Cabin", axis=1, inplace=True)
    data.dropna(inplace=True)
    sex = pd.get_dummies(data["Sex"], drop_first=True)
    embark = pd.get_dummies(data["Embarked"], drop_first=True)

    data = pd.concat([data, sex, embark], axis=1)

    data.drop(["Sex", "Embarked", "Name", "Ticket", "PassengerId"], axis=1, inplace=True)

    plt.figure()
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False)

    return data

def learning(data):
    x = data.drop("Survived", axis=1)
    y = data["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
    logmodel = LogisticRegression()
    logmodel.fit(x_train, y_train)

    predictions = logmodel.predict(x_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

train = pd.read_csv("Logistic-Regression/titanic_train.csv")
test = pd.read_csv("Logistic-Regression/titanic_test.csv")

#initial_analysis(train, "Training Data")
train = cleaner(train)
learning(train)

plt.show()