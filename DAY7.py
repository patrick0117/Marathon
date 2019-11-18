# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:23:14 2019

@author: 91812
知道 DataFrame 如何檢視欄位的型態數量以及各欄型態, 以及 Label Encoding / One Hot Encoding 如何寫?
檢視 DataFrame 的資料型態 (In[3], In[4])
了解 Label Encoding 如何寫 (In[6])
了解 One Hot Encoding 如何寫 (In[7])
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  
#讓同一個欄位底下的類別之間有大小關係 (0<1<2)

dir_data = 'C:/Users/91812/Documents/GitHub/Marathon/downloads'
f_app_train = os.path.join(dir_data, 'D6_train.csv')
f_app_test = os.path.join(dir_data, 'D6.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

#app_train.dtypes.value_counts()    可顯示內部資料欄位類型的數量
#app_train.select_dtypes(include=["object"]).apply(pd.Series.nunique, axis = 0)  可顯示資料中類別型欄位各自類別的數量

#LABLE ENCODING
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col]) 
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

#one hot
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print(app_train['CODE_GENDER_F'].head())
print(app_train['CODE_GENDER_M'].head())
print(app_train['NAME_EDUCATION_TYPE_Academic degree'].head())

#HW
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()
