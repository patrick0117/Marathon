# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:09:04 2019
以下用房價預測資料, 觀察降低偏態的影響
觀察原始數值的散佈圖, 以及線性迴歸分數 (Out[4], Out[5])
觀察使用log1p降偏態時, 對於分布與迴歸分數的影響 (In[6], Out[6])
觀察使用box-cox(λ=0.15)時, 對於分布與迴歸分數的影響 (In[7], Out[7])
觀察使用sqrt(box-cox, λ=0.5)時, 對於分布與迴歸分數的影響 (In[8], Out[8])
@author: 91812
"""
# 做完特徵工程前的所有準備
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

data_path = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'
df_train = pd.read_csv(data_path + 'house_train.csv.gz')
df_test = pd.read_csv(data_path + 'house_test.csv.gz')

train_Y = np.log1p(df_train['SalePrice'])
ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)  #移除欄位
df_test = df_test.drop(['Id'] , axis=1)   #移除欄位
df = pd.concat([df_train,df_test]) #concat合併資料
# df.head()

# 只取 int64,float64兩種欄位，存於num feature中
num_features=[]
for dtype, feature in zip(df.dtype, df.columns):
    if dtype =='float64' or dtype =='int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')
# f string {變數} \n換行

# 削減文字欄位, 只剩數值型攔位
df = df[num_features]
df= df.fillna(-1)
MMEncoder = MinMaxScaler()
train_num = train_Y.shape[0]
df.head()

# 顯示lotarea散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df['lotArea'[:train_num]])
plt.show()

# 計算基礎分數
df_mm = MMEncoder.fit_transform(df)
train_X = df_mm[:train_num]
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

# 1.將lotarea 取log1p後, 看散佈圖, 並計算分數
df_fixed = copy.deepcopy(df)
df_fixed['LotArea']=np.log1p(df_fixed['LotArea'])
# 當數值很小時 用log1p答案不會為0 會是一個很小的數字
sns.displot(df_fixed['LotArea'][:train_num])
plt.show()

df_fixed = MMEncoder.fit_transform(df_fixed)
train_X = df_fixed[:train_num]
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

# 2. 將LotArea取boxcox後 看散佈圖, 並計算分數
from scipy import stats
df_fixed = copy.deepcopy(df)
df_fixed['LotArea'] = stats.boxcox(df_fixed['LotArea'],lmbda=0.15)
sns.displot(df_fixed['LotArea'][:train_num])
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

# 3. 將 LotArea 取 sqrt (box-cox : alpha=0.5) 後, 看散佈圖, 並計算分數
df_fixed = copy.deepcopy(df)
df_fixed['LotArea'] = stats.boxcox(df['LotArea'], lmbda=0.5)
sns.distplot(df_fixed['LotArea'][:train_num])
plt.show()

df_fixed = MMEncoder.fit_transform(df_fixed)
train_X = df_fixed[:train_num]
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

