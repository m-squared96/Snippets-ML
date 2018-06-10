#!/usr/bin/python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

def explore(df):
    print(df.columns)
    print('\n')
    print(df.head())

    plot_option = str(input("Would you like to visualise the raw data?(y/n)   "))

    if plot_option.lower() == "y" or plot_option.lower() == "yes":
        visualise(df)

def visualise(df):
    plt.figure()
    sns.pairplot(data, hue="species")

    plt.figure()
    sns.kdeplot(data[data['species'] == 'setosa']['sepal_width'],data[data['species'] == 'setosa']['sepal_length'], shade=True, cmap="Purples")
    plt.title("Setosa Sepal Widths and Lengths")

def model_train(df):
    xtr, xte, ytr, yte = train_test_split(df.drop('species', axis=1),df['species'],test_size=0.5, random_state=101)
    model_list = [xtr,xte,ytr,yte]
    model = SVC()
    model.fit(xtr,ytr)
    pred = model.predict(xte)

    evaluate(yte,pred)
    grid_train(model_list)

def grid_train(model_list):
    param_grid = {'C':[0.1,1,10,100,1000,10000,100000], 'gamma':[1,0.1,0.01,0.001,0.0001,0.00001], 'kernel':['rbf']}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
    grid.fit(model_list[0],model_list[2])
    grid_pred = grid.predict(model_list[1])

    evaluate(model_list[3],grid_pred)
    
def evaluate(test,predictions):
    print(confusion_matrix(test, predictions))
    print('\n')
    print(classification_report(test, predictions))


data = sns.load_dataset('iris')
explore(data)
model_train(data)

plt.show()
