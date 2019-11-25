# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:46:33 2019

@author: 91812
"""

# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 設定 data_path
dir_data = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'

f_app = os.path.join(dir_data, 'D6_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)

#-----------------------------------ex---------------------------------------------------#
# 先篩選數值型的欄位
#
#print((app_train['DAYS_BIRTH'] / (-365)).describe())

#(app_train['DAYS_EMPLOYED'] / 365).describe()
plt.hist(app_train['DAYS_EMPLOYED'])
plt.show()
app_train['DAYS_EMPLOYED'].value_counts()
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243] 
#去掉過大的數值
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))
#-----------------------------------------------------------------------------------------#
#想知道怎麼一次檢視所有的欄位
#---------------------------------hw---------------------------------------------#
dtype_select = []
#在app train 內選出某幾欄
#篩選數值欄位                    #Column內               #數字形式      #整數、浮點數           
numeric_columns = list(app_train.columns[list(app_train.dtypes.isin([np.int64, np.float64]))])
# 把只有 2 值 (通常是 0,1) 的欄位去掉 !=2不取種類2種的                                                  #長度(資料種類)不等於2
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
print("Numbers of remain columns: %i" % len(numeric_columns))

# 檢視這些欄位的數值範圍
"""
for col in numeric_columns:
    app_train.boxplot(col)
    plt.show()
"""
a = app_train['OWN_CAR_AGE'].dropna()
plt.hist(a)

# 從上面的圖檢查的結果，至少這三個欄位好像有點可疑

# AMT_INCOME_TOTAL
# REGION_POPULATION_RELATIVE
# OBS_60_CNT_SOCIAL_CIRCLE

 # 最大值離平均與中位數很遠
print(app_train['AMT_INCOME_TOTAL'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
#index編號   cumsum累加   sort 分類可以对 index 進行排序     
cdf = app_train.AMT_INCOME_TOTAL.value_counts().sort_index().cumsum()
plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) # 限制顯示圖片的範圍
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

plt.plot(np.log(list(cdf.index)), cdf/cdf.max())
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')

plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

#####----------------------------------------補充----------------------------------------------#####
# 最大值落在分布之外
print(app_train['REGION_POPULATION_RELATIVE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
cdf = app_train.REGION_POPULATION_RELATIVE.value_counts().sort_index().cumsum()
plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()

app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()

app_train['REGION_POPULATION_RELATIVE'].value_counts()

# 就以這個欄位來說，雖然有資料掉在分布以外，也不算異常，僅代表這間公司在稍微熱鬧的地區有的據點較少，
# 導致 region population relative 在少的部分較為密集，但在大的部分較為疏漏

# 最大值落在分布之外
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
cdf = app_train.OBS_60_CNT_SOCIAL_CIRCLE.value_counts().sort_index().cumsum()
plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min() * 0.95, cdf.index.max() * 1.05])
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()

app_train['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()

app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index(ascending = False)







