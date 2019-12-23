# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:55:59 2019
使用擬合過的模型, 計算特徵重要性 (In[4], Out[4])
對照原始特徵, 觀察特徵重要性較高的一半特徵, 搭配隨機森林對於測結果的影響 (In[5], Out[5], In[6], Out[6])
重組重要性最高的特徵作為新特徵, 觀察效果如何 (In[9], Out[9])
@author: 91812
"""
# 做完特徵工程前的所有準備
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

data_path = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'
df = pd.read_csv(data_path + 'house_train.csv.gz')

train_Y = np.log1p(df['SalePrice'])#目標是預測價錢
df = df.drop(['Id'] , axis=1)
#df.head()

# 因為需要把類別型與數值型特徵都加入, 故使用最簡版的特徵工程
LEncoder = LabelEncoder()    #1,2,3,4,5,6,7,8,9,10......
MMEncoder = MinMaxScaler()   #標準化縮放特徵
for c in df.columns:
    df[c] = df[c].fillna(-1) #自動填補空值 auto fill empty side
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
        #fit : method calculates internal parameters
        #transform : method using the calculated parameters apply hte transform to particular dataset
        #fit_transform : joined fit and transform together
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
    #reshape into range -1 to +1

# 隨機森林擬合後, 將結果依照重要性由高到低排序
estimator = RandomForestRegressor()
estimator.fit(df.values, train_Y)
# estimator.feature_importances_ 就是模型的特徵重要性
# 這邊先與合起來, 才能看到重要性與欄位名稱的對照表
feats = pd.Series(data=estimator.feature_importances_, index=df.columns) 
       #pd.series : create a series by input list
       #create a column called "feats" with data inside
feats = feats.sort_values(ascending=False)      
        #sort_value : arrange by column or row
        #ascending = 上升   ascending=False : form high to low
#feats
    
# 原始特徵 + 隨機森林
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()#cross validate 5 times

# 高重要性特徵 + 隨機森林 (39大約是79的一半)
high_feature = list(feats[:39].index)
train_X = MMEncoder.fit_transform(df[high_feature])
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
    
    # 觀察重要特徵與目標的分布
# 第一名 : OverallQual              
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x=df['OverallQual'], y=train_Y, fit_reg=False)
plt.show()
    
    # 第二名 : GrLivArea
sns.regplot(x=df['GrLivArea'], y=train_Y, fit_reg=False)
plt.show()
    
   # 製作四特徵 : 加, 乘, 互除(分母加1避免除0) 看效果 (Note: 數值原本已經最大最小化介於 [0,1] 區間, 這四種新特徵也會落在 [0,1] 區間)
df['Add_char'] = (df['GrLivArea'] + df['OverallQual']) / 2
df['Multi_char'] = df['GrLivArea'] * df['OverallQual']
df['GO_div1p'] = df['GrLivArea'] / (df['OverallQual']+1) * 2
df['OG_div1p'] = df['OverallQual'] / (df['GrLivArea']+1) * 2
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    