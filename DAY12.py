import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# 讀取訓練與測試資料
data_path = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'
df_train = pd.read_csv(data_path + 'house_train.csv.gz')
df_test = pd.read_csv(data_path + 'house_test.csv.gz')

# 重組資料成為訓練 / 預測用格式
train_Y = np.log1p(df_train['SalePrice'])
ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)
df = pd.concat([df_train,df_test])
#檢查缺失數量 空缺.相加.的累積
#df.isnull().sum().sort_values(ascending=False).head()

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
train_num = train_Y.shape[0]

# 空值補 -1
df_m1 = df.fillna(-1)
train_X = df_m1[:train_num]
# 空值補 0
df_0 = df.fillna(0)
train_X = df_0[:train_num]
# 空值補平均
df_mn = df.fillna(df.mean())
train_X = df_mn[:train_num]
# 空值補 -1, 搭配最大最小化
df = df.fillna(-1)
df_temp = MinMaxScaler().fit_transform(df)
train_X = df_temp[:train_num]
# 搭配標準化
df_temp = StandardScaler().fit_transform(df)
train_X = df_temp[:train_num]
#做線性迴歸
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

#HW1
# 空值補 -1, 做羅吉斯迴歸

df_hw11 = df.fillna(-1)
train_X = df_hw11[:train_num]
hw11=cross_val_score(estimator, train_X, train_Y, cv=5).mean()


df_hw12 = df.fillna(100)
train_XX = df_hw12[:train_num]
hw12=cross_val_score(estimator, train_XX, train_Y, cv=5).mean()





































