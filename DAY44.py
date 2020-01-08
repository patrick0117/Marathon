# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:16:59 2020

@author: tribology2020
"""
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)
clf = RandomForestClassifier(n_estimators=30, #number of trees
                             criterion="gini",
                             max_features="auto", #how to choose feature
                             max_depth=4,
                             min_samples_split=2,
                             min_samples_leaf=1
                             )

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

print(iris.feature_names)

print("Feature importance: ", clf.feature_importances_)











