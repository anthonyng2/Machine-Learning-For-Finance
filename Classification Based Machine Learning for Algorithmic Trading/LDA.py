# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:51:39 2017

@author: AnthonyN
"""

import numpy as np
import pandas as pd
lags = 5
start_test = pd.to_datetime('2017-06-18')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ts = pd.read_csv('XMA.csv', index_col='Date')
ts.index = pd.to_datetime(ts.index)
tslag = ts[['XMA']].copy()

for i in range(0,lags):
    tslag["Lag_" + str(i+1)] = tslag["XMA"].shift(i+1)
tslag["returns"] = tslag["XMA"].pct_change()
tslag.dropna(inplace=True)

# Create the "Direction" column (+1 or -1) indicating an up/down day
tslag["Direction"] = np.sign(tslag["returns"])

# Use the prior two days of returns as predictor values, with direction as the response
X = tslag[["Lag_1","Lag_2"]]
y = tslag["Direction"]

# Create training and test sets
X_train = X[X.index < start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]

# Create prediction DataFrame
pred = pd.DataFrame(index=y_test.index)

lda = LDA()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
pred = (1.0 + y_pred * y_test)/2.0
hit_rate = np.mean(pred)
print('LDA {:.3f}'.format(hit_rate))