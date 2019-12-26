# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 16:11:58 2019
學習使用 sklearn 中的 train_test_split 等套件，進行資料的切分
@author: 91812
"""
from sklearn.model_selection import train_test_split, KFold
import numpy as np
X = np.arange(50).reshape(10, 5) # 生成從 0 到 50 的 array，並 reshape 成 (10, 5) 的 matrix
y = np.zeros(10) # 生成一個全零 arrary
y[:5] = 1 # 將一半的值0~4改為 1
print("Shape of X: ", X.shape)
print("Shape of y: ", y.shape)

print('X: shape: ' + str(X.shape))
print(X)
print("")
print('y: shape: ' + str(y.shape))
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#使用 K-fold Cross-validation 來切分資料
kf = KFold(n_splits=5)
i = 0
for train_index, test_index in kf.split(X):
    i +=1 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("FOLD {}: ".format(i))
    print("X_test: ", X_test)
    print("Y_test: ", y_test)
    print("-"*30)
#HW
X = np.arange(1000).reshape(200, 5)
y = np.zeros(200)
y[:40] = 1
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X):
    i +=1 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("FOLD {}: ".format(i))
    print("X_test: ", X_test)
    print("Y_test: ", y_test)
    print("-"*30)
#COMBINE y_test &y_train
