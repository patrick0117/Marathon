# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:18:20 2019
計算並觀察百分位數 (In[4], In[7])
計算中位數的方式 (In[8])
計算眾數 (In[9], In[10])
計算標準化與最大最小化 (In[11])

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import time

# 設定 data_path
dir_data = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'
f_app = os.path.join(dir_data, 'D6_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)

#如果欄位中有 NA, describe 會有問題
app_train['AMT_ANNUITY'].describe()

# Ignore NA, 計算五值   percentile代表第幾個百分比的分位數    isnull判斷缺失
five_num = [0, 25, 50, 75, 100]
quantile_5s = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in five_num]
print(quantile_5s)

app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'].hist(bins = 100)
plt.show()

# 試著將 max 取代為 q99
app_train[app_train['AMT_ANNUITY'] == app_train['AMT_ANNUITY'].max()] = np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = 99)

# 得到 median 的另外一種方法
np.median(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])

# 計算眾數 (mode)
start_time = time.time()
mode_get = mode(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])
print(mode_get)
print("Elapsed time: %.3f secs" % (time.time() - start_time))

# 計算眾數 (mode)
# 較快速的方式
from collections import defaultdict

start_time = time.time()
mode_dict = defaultdict(lambda:0)

for value in app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY']:
    mode_dict[value] += 1
    
mode_get = sorted(mode_dict.items(), key=lambda kv: kv[1], reverse=True)
print(mode_get[0])
print("Elapsed time: %.3f secs" % (time.time() - start_time))

#連續值標準化
'''
1. Z-transform: $ \frac{(x - mean(x))}{std(x)} $
2. Range (0 ~ 1): $ \frac{x - min(x)}{max(x) - min(x)} $
3. Range (-1 ~ 1): $ (\frac{x - min(x)}{max(x) - min(x)} - 0.5) * 2 $
'''

app_train['AMT_CREDIT'].hist(bins = 50)
plt.title("Original")
plt.show()
value = app_train['AMT_CREDIT'].values

app_train['AMT_CREDIT_Norm1'] = ( value - np.mean(value) ) / ( np.std(value) )
app_train['AMT_CREDIT_Norm1'].hist(bins = 50)
plt.title("Normalized with Z-transform")
plt.show()

app_train['AMT_CREDIT_Norm2'] = ( value - min(value) ) / ( max(value) - min(value) )
app_train['AMT_CREDIT_Norm2'].hist(bins = 50)
plt.title("Normalized to 0 ~ 1")
plt.show()

#HW
'''
計算 AMT_ANNUITY 的分位點 (q0 - q100) (Hint : np.percentile, In[3])
將 AMT_ANNUITY 的 NaN 用中位數取代 (Hint : q50, In[4])
將 AMT_ANNUITY 數值轉換到 -1 ~ 1 之間 (Hint : 參考範例, In[5])
將 AMT_GOOD_PRICE 的 NaN 用眾數取代 (In[6])
'''
#計算 AMT_ANNUITY 的分位點 (q0 - q100)
hundred_num = np.linspace(0,100,100)
quantile_100s = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in five_num]
print(quantile_100s)

#將 AMT_ANNUITY 的 NaN 用中位數取代 
app_train[app_train['AMT_ANNUITY'] == app_train['AMT_ANNUITY'].isnull()] = np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = 50)

#將 AMT_ANNUITY 數值轉換到 -1 ~ 1 之間
def normalize_value(x):
    """
    Your Code Here, compelete this function
    """
    
    return x

app_train['AMT_ANNUITY_NORMALIZED'] = normalize_value(app_train['AMT_ANNUITY'])

print("== Normalized data range ==")
app_train['AMT_ANNUITY_NORMALIZED'].describe()

#將 AMT_GOOD_PRICE 的 NaN 用眾數取代 (In[6])
print("Before replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))

# 列出重複最多的數值
"""
Your Code Here
"""
value_most = mode(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])
print(value_most)

mode_goods_price = list(app_train['AMT_GOODS_PRICE'].value_counts().index)
app_train.loc[app_train['AMT_GOODS_PRICE'].isnull(), 'AMT_GOODS_PRICE'] = mode_goods_price[0]

print("After replace NAs, numbers of row that AMT_GOODS_PRICE is NAs: %i" % sum(app_train['AMT_GOODS_PRICE'].isnull()))






















