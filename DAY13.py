# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:59:41 2019

@author: 91812
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 生成範例用的資料 ()
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])

# 沿縱軸合併
result_v = pd.concat([df1, df2, df3])
# 沿橫軸合併
result_h = pd.concat([df1, df4], axis = 1)

#沿橫軸合併
resulth1 = pd.concat([df1, df4], axis = 1, join = 'inner') # 硬串接
resulth2 = pd.merge(df1, df4, how='inner')

# 將 欄-列 逐一解開
print(df1)
#df1.melt()

#subset子集
dir_data = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
# 取 TARGET 為 1 的
sub_df = app_train[app_train['TARGET'] == 1]
#sub_df.head()

# 取 AMT_INCOME_TOTAL 大於平均資料中，SK_ID_CURR, TARGET 兩欄
sub_df = app_train.loc[app_train['AMT_INCOME_TOTAL'] > app_train['AMT_INCOME_TOTAL'].mean(), ['SK_ID_CURR', 'TARGET']]
sub_df.head()

#Groupby分組
#app_train.groupby(['NAME_CONTRACT_TYPE']).size()
#app_train.groupby(['NAME_CONTRACT_TYPE'])['AMT_INCOME_TOTAL'].describe()
#app_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].mean()

# 取前 10000 筆作範例: 分別將 AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY 除以根據 NAME_CONTRACT_TYPE 分組後的平均數，
#app_train.loc[0:10000, ['NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']].groupby(['NAME_CONTRACT_TYPE']).apply(lambda x: x / x.mean())

app_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].hist()
plt.show()
'''
HW
1.請將 app_train 中的 CNT_CHILDREN 依照下列規則分為四組，並將其結果在原本的 dataframe 命名為 CNT_CHILDREN_GROUP
  0 個小孩、有 1 - 2 個小孩、有 3 - 5 個小孩、有超過 5 個小孩
2.請根據 CNT_CHILDREN_GROUP 以及 TARGET，列出各組的平均 AMT_INCOME_TOTAL，並繪製 baxplot

3.請根據 CNT_CHILDREN_GROUP 以及 TARGET，對 AMT_INCOME_TOTAL 計算 Z 轉換 後的分數
'''
#1.
cut_rule = [-np.inf, 0, 2, 5, np.inf]
#pd.cut(group want to cut, cut rule)
app_train['CNT_CHILDREN_GROUP'] = pd.cut(app_train['CNT_CHILDREN'].values, cut_rule, include_lowest=True)
#上列切分就是在給不同年齡一個分組代號，下面就是計算各組組數
value1=app_train['CNT_CHILDREN_GROUP'].value_counts()
#2.
#創建一list-grp
grp = ['CNT_CHILDREN_GROUP', 'TARGET']
#使用groupby()方法可以將資料依照自己要的column分組
#讓AMT_INCOME_TOTAL依照(grp)分組
grouped_df = app_train.groupby(grp)['AMT_INCOME_TOTAL']

#grouped_df.mean()

plt_column = 'AMT_INCOME_TOTAL'
plt_by = ['CNT_CHILDREN_GROUP', 'TARGET']

app_train.boxplot(column=plt_column, by = plt_by, showfliers = False, figsize=(12,12))
plt.suptitle('')
plt.show()
#3.
app_train['AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET'] = grouped_df.apply(lambda x:(x-np.mean(x))/np.std(x))

























