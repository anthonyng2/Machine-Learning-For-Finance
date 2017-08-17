# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:02:07 2017

@author: Anthony
"""
import numpy as np
import pandas as pd
df = pd.read_csv("dataset_2.csv")

df['default'].describe()
sum(df['default'] == 0)
sum(df['default'] == 1)

X = df.iloc[:, 1:6].values
y = df['default'].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=0)

shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn import gaussian_process
clf = gaussian_process.GaussianProcessClassifier(random_state=0)
clf.fit(X_train, y_train)

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
cm = confusion_matrix(y_train, y_train_pred)
print(cm)

from sklearn.metrics import precision_score, recall_score
print("precision score = {0:.4f}".format(precision_score(y_train, y_train_pred)))
print("recall score =  {0:.4f}".format(recall_score(y_train, y_train_pred)))
