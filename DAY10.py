# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:06:30 2019

@author: 91812
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


data_path = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'
df_train = pd.read_csv(data_path + 'D7house_train.csv.gz')

train_Y = np.log1p(df_train['SalePrice'])
df = df_train.drop(['Id', 'SalePrice'] , axis=1)

num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

df = df[num_features]
#使用指定的方法填充NA  NA轉變為-1
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
#regplot回歸模型繪圖
sns.regplot(x = df['GrLivArea'], y=train_Y)
plt.show()

# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
#1.
# 將 GrLivArea 限制在 800 到 2500 以內, 調整離群值
df['GrLivArea'] = df['GrLivArea'].clip(400, 3200)
sns.regplot(x = df['GrLivArea'], y=train_Y)
plt.show()
# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

#2.
# 將 GrLivArea 限制在 800 到 2500 以內, 捨棄離群值
keep_indexs = (df['GrLivArea']> 400) & (df['GrLivArea']<3200)
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]
sns.regplot(x = df['GrLivArea'], y=train_Y)
plt.show()
# 做線性迴歸, 觀察分數
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())














