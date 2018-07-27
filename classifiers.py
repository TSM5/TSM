# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:57:40 2018

@author: manik
"""

'''this project is using the all available classification algorithms both  linear and non linear 
algorithms to predict breast cancer data available from the uci repo'''
#import the libraries
import numpy as np
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
filename = 'breast-cancer-wisconsin.data'
names = ['Sample code number ', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin','Normal Nucleoli','Mitoses' ,'class']
dataframe = read_csv(filename, names=names,delimiter=',')
#knowing datatypes of each columns
dataframe.dtypes
#finding missing values
#convert missing values from ? to NUMPY NAN
df=dataframe.replace('?',np.NaN)
#counting number of instances missing with each column
df.isna().sum()
#Now we came to know that bare nuclei is the one having missing values 
#implementing missing values
df=df.fillna(method='ffill')
#knowing from data
data=df.describe()
#class counts
df.groupby('class').size()
#correlations between them
core=df.corr()
#skewness of univariate distributions
skew=df.skew()
#visualizing data 
#histograms
df.hist()
#density plots
df.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
#here we can see all of them are right skewed so we can use log to convert to normal
#boxplots
data.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False) 
#correlation graph can also be done
#scatter plots
pd.tools.plotting.scatter_matrix(df)
#seperating input and output as two different variables
X=df.iloc[:,1:10].values
Y=df.iloc[:,10].values
#here I have taken the default constructors
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
  kfold = KFold(n_splits=10, random_state=7)
  cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
