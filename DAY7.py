# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:25:46 2019

@author: 91812
如何觀察目前的 DataFrame 中, 有哪些欄位類型, 以及數量各有多少 (In[3], Out[3])
如何將欄位名稱依欄位類型分開 (In[4], Out[4])
如何只顯示特定類型的欄位資料 (In[5], Out[5])
"""
# 載入基本套件
import pandas as pd
import numpy as np

# 讀取訓練與測試資料
data_path = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'
df_train = pd.read_csv(data_path + 'D7house_train.csv.gz')
df_test = pd.read_csv(data_path + 'D7house_test.csv.gz')
print(df_train.shape)

# 訓練資料需要 train_X, train_Y / 預測輸出需要 ids(識別每個預測值), test_X
# 在此先抽離出 train_Y 與 ids, 而先將 train_X, test_X 該有的資料合併成 df, 先作特徵工程
train_Y = np.log1p(df_train['SalePrice'])
ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)   #去掉拿出來的段落:drop
df_test = df_test.drop(['Id'] , axis=1)                  #axis=1 直行
df = pd.concat([df_train,df_test])                       #concat合併資料
print(df.head())

# 秀出資料欄位的類型, 與對應的數量
# df.dtypes : 轉成以欄位為 index, 類別(type)為 value 的 DataFrame
# .reset_index() : 預設是將原本的 index 轉成一個新的欄位, 如果不須保留 index, 則通常會寫成 .reset_index(drop=True)
dtype_df = df.dtypes.reset_index() 
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
print('Data type:',dtype_df)

# 確定只有 int64, float64, object 三種類型後對欄位名稱執行迴圈, 分別將欄位名稱存於三個 list 中
int_features = []
float_features = []
object_features = []
# .dtypes(欄位類型), .columns(欄位名稱) 是 DataFrame 提供的兩個方法, 這裡順便展示一下 for 與 zip 搭配的用法
#dtype 是來自df.dtypes 的類型，feature是來自df.columns 的欄位名稱
#從26行取出type 與 columnzp 分別一對一配對起來
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)
#完成後可以將三種不同資料型態分類(integer,float,object)
# 這邊採用的寫法稱為 f-string, 是 Python 3.6.2 以後版本才出現的
# 如果無法執行, 則需要更新到這個版本之後, 或自行將程式改寫為 str.format 形式
# 改寫方式可以參考 https://blog.louie.lu/2017/08/08/outdate-python-string-format-and-fstring/
print(f'{len(int_features)} Integer Features : {int_features}\n')
print(f'{len(float_features)} Float Features : {float_features}\n')
print(f'{len(object_features)} Object Features : {object_features}')
#f-string : Literal String Interpolation

#HW
'''
1.
試著執行作業程式，觀察三種類型的欄位分別進行( 平均 mean / 最大值 Max / 相異值 nunique ) 中的九次操作會有那些問題?
並試著解釋那些發生Error的程式區塊的原因?
'''

'''
int_mean=df[int_features].mean()        #算出平均值ok!
int_max=df[int_features].max()          #找出了最大值ok!
int_nunique=df[int_features].nunique()  #算出內部獨立個數的數量>>有幾種資料 

float_mean=df[float_features].mean()
float_max=[float_features].max()        #list沒有最大值
float_nunique=df[float_features].nunique()

object_mean=df[object_features].mean()  #字串沒答案
object_max=[object_features].max()      #沒
object_nunique=df[object_features].nunique() #成功
'''

'''
2.
思考一下，試著舉出今天五種類型以外的一種或多種資料類型，你舉出的新類型是否可以歸在三大類中的某些大類?
所以三大類特徵中，哪一大類處理起來應該最複雜?
dict字典(容器)、tuple元組(容器)、chr字元(字,組成字串)、bool布林(數值,010101)
容器類比較難處理八，因為可以裝不同種類的資料?

'''






